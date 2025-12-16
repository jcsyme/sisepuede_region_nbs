import numpy as np
import pandas as pd
import pathlib
import sisepuede.core.information_table_properties as sitp
import sisepuede.core.model_attributes as ma
import sisepuede.utilities._toolbox as sf
from typing import *


##########################
#    GLOBAL VARIABLES    #
##########################

# fields
_FIELD_CW_IEA_FUELS = "iea_fuels"
_FIELD_CW_IEA_FUEL_GROUPS = "iea_fuel_groups"
_FIELD_IEA_FUEL = "electricity generation sources in China"
_FIELD_IEA_UNITS = "Units"
_FIELD_IEA_VALUE = "Value"
_FIELD_IEA_YEAR = "Year"
_FIELD_PREFIX_TOTAL = "total"

# file name components
_FNC_IEA_COM = "commercial and public services"
_FNC_IEA_IND = "industry"
_FNC_IEA_RES = "residential"
_FNC_IEA_TRS = "transport"

# information table from SISEPUEDE
_ITP = sitp.InformationTableProperties()


def build_allocation_dict(
    df_input: pd.DataFrame,
    models: 'SISEPUEDEModels',
    df_out_preliminary: Union[pd.DataFrame, None] = None,
    df_out_preliminary_no_cm: Union[pd.DataFrame, None] = None,
    df_out_preliminary_no_re: Union[pd.DataFrame, None] = None,
    **kwargs,
) -> Dict[str, Tuple]:
    """Build a dictionary used for allocation and iterative fuel
        shifting.

    Maps each subsec to tuple with following ordered elements:
        - SSP output dataframe to use for calibration scaling
        - SSP ModelVariable
        - any category restrictions

        
    Function Arguments
    ------------------
    df_input : pd.DataFrame
        Input DataFrame
    models : SISEPUEDEModels
        Used for variable access and management

        
    Keyword Arguments
    -----------------
    df_out_preliminary: Union[pd.DataFrame, None]
        Optional passing of existing df_out_prelim (for calib)
    df_out_preliminary_no_cm_cur: Union[pd.DataFrame, None]
    df_out_preliminary_no_re_cur: Union[pd.DataFrame, None]
    **kwargs :
        Passed get_enercons_calib_dfs()
    """

    ##  INITIALIZATION AND SHORTCUTS
    
    model = models.model_enercons
    matt = model.model_attributes

    # rerun the models (no enerprod)?
    rerun = not isinstance(df_out_preliminary, pd.DataFrame)
    rerun |= not isinstance(df_out_preliminary_no_cm, pd.DataFrame)
    rerun |= not isinstance(df_out_preliminary_no_re, pd.DataFrame)
    if rerun:
        (
            df_out_preliminary,
            df_out_preliminary_no_cm,
            df_out_preliminary_no_re,
        ) = get_enercons_calib_dfs(
            df_input, 
            models,
            **kwargs,
        )
    
    # get model variables
    modvar_inen = matt.get_variable(model.modvar_enfu_energy_demand_by_fuel_inen, )
    modvar_scoe = matt.get_variable(model.modvar_enfu_energy_demand_by_fuel_scoe, )
    modvar_trns = matt.get_variable(model.modvar_enfu_energy_demand_by_fuel_trns, )
    
    # build output dictionary
    dict_out = {
        _FNC_IEA_COM : (
            df_out_preliminary_no_re,
            modvar_scoe,
            ["commercial_municipal", "other_se"],
        ),
    
        _FNC_IEA_IND: (
            df_out_preliminary,
            modvar_inen,
            None,
        ),
    
        _FNC_IEA_RES: (
            df_out_preliminary_no_cm,
            modvar_scoe,
            ["residential"],
        ),
        
        _FNC_IEA_TRS: (
            df_out_preliminary,
            modvar_trns,
            None,
        ),
    
    }

    return dict_out



def field_iea_fuel_from_path(
    path: pathlib.Path,
) -> str:
    """Get the IEA fuel field from the path name. Works for consumption
    """
    field = (
        path
        .parts[-1]
        .split(" - ")[1]
        .replace(".csv", "")
    )

    return field
    


def get_and_allocate_fuels_from_ssp(
    path_iea_cons: pathlib.Path,
    path_iea_cats: pathlib.Path,
    df_ssp_allocation_out: pd.DataFrame,
    modvar: Union['ModelVariable', str],
    models: 'SISEPUEDEModels',
    time_periods: 'TimePeriods',
    df_varinfo: Union[pd.DataFrame, None] = None,
    field_cat_value: str = _ITP.field_categories,
    field_iea_year: str = _FIELD_IEA_YEAR,
    field_variable: str = _ITP.field_field_emission,
    groups_allocate: List[str] = ["Oil products"],
) -> Tuple[pd.DataFrame]:
    """Some IEA consumptions uses "oil products" for consumption without
        specifying how those oil products are allocated. This function 
        uses SSP input fractions as a baseline to allocate the IEA 
        categories.

    Returns a tuple of DataFrames of (a) estimated consumption by fuel and 
        (b) estimated fraction of consumption by fuel, i.e., 

        (
            df_consumption,
            df_consumption_fraction,
        )

    Function Arguments
    ------------------
    path_iea_cons : pathlib.Path
        Path to IEA consumption file to allocate
    path_iea_cats : pathlib.Path
        Path to IEA categories crosswalk file
    df_ssp_allocation : pd.DataFrame
        DataFrame used to allocate
    modvar : Union['ModelVariable', str]
        ModelVariable that is pulled from to estimate average consumption for each 
        fuel
    models : SISEPUEDEModels
        SISEPUEDEModels object used to access variables/manage through associated
        ModelAttributes object
    time_periods : TimePeriods
        Used for time period management and conversion to and from years
        
    Keyword Arguments
    -----------------
    df_varinfo : Union[pd.DataFrame, None]
        Optional DataFrame storing the variable info
    field_iea_year : Union[str]
        Field in IEA consumption data storing the year 
    groups_allocate : List[str]
        Groups to allocate to SSP fuels
    """


    ##  INITIALIZATION

    # model attributes objects
    matt = models.model_attributes
    attr_enfu = matt.get_attribute_table(
        matt.subsec_name_enfu,
    )
    field_time_period = matt.dim_time_period

    # check the variable info dictionary
    df_varinfo = (
        sitp.build_variable_information_table(matt, None, )
        if not isinstance(df_varinfo, pd.DataFrame)
        else df_varinfo
    )
    
    # get grouping
    df_iea_cw = pd.read_csv(path_iea_cats, )
    
    _, dict_iea_fuels_to_ssp_fuels = get_iea_dict(
        df_iea_cw, 
        models.model_attributes, 
        dict_type = "iea_fuels",
    )
    
    _, dict_iea_fuel_groups_to_ssp_fuels = get_iea_dict(
        df_iea_cw, 
        models.model_attributes, 
        dict_type = "iea_fuel_groups",
    )

    
    # get IEA data
    df_iea = pd.read_csv(path_iea_cons, )
    field_iea_fuel = field_iea_fuel_from_path(path_iea_cons, )
    
    df_iea, _, _, _ = get_iea_df_for_ssp(
        df_iea,
        dict_iea_fuels_to_ssp_fuels,
        time_periods,
        models,
        field_iea_fuel,
        convert_fuel_to_tech = False,
        dict_category_replacements = dict(zip(attr_enfu.key_values, attr_enfu.key_values,)),
        #modvar = _MODVAR_MSP,
        fuels_iea_keep = groups_allocate,
        normalize = False,
    ) 

    
    # iterate over each group
    for grp in groups_allocate:

        if not grp in df_iea.columns: continue
            
        # try to get fuels
        fuels_to_allocate_to = dict_iea_fuel_groups_to_ssp_fuels.get(grp, )
        
        # checks
        if not sf.islistlike(fuels_to_allocate_to): continue
        fuels_to_allocate_to = [x for x in attr_enfu.key_values if x in fuels_to_allocate_to]
        if len(fuels_to_allocate_to) == 0: continue
        

        # get allocation fields, then pull from output data and normalize to one
        fields_alloc = [
            x for x in modvar.build_fields(category_restrictions = fuels_to_allocate_to, ) 
            if x is not None
        ]

        # 
        dict_rnm = dict(zip(fields_alloc, fuels_to_allocate_to, ))
        df_allocate = df_ssp_allocation_out[[field_time_period] + fields_alloc]
        df_allocate[fields_alloc] = sf.check_row_sums(
            df_allocate[fields_alloc],
            thresh_correction = None,
        )

        # merge in to iea data, then distribute the original group across
        df_iea = (
            pd.merge(
                df_iea,
                df_allocate
                .rename(
                    columns = {time_periods.field_year: field_iea_year, }
                ),
                how = "left",
            )
            .interpolate()
            .ffill()
            .bfill()
        )

        df_iea[fields_alloc] = sf.do_array_mult(
            df_iea[fields_alloc].to_numpy(),
            df_iea[grp].to_numpy()
        )

        df_iea = (
            df_iea
            .drop(columns = grp)
            .rename(columns = dict_rnm, )
        )


    fields_ind = [time_periods.field_time_period]
    fields_dat = sorted(
        [
            x for x in df_iea.columns if x not in fields_ind
        ]
    )

    df_iea = df_iea.get(fields_ind + fields_dat)
    df_iea[fields_ind] = df_iea[fields_ind].astype(int)


    # finally, get a normalized version
    df_iea_norm = df_iea.copy()
    df_iea_norm[fields_dat] = sf.check_row_sums(
        df_iea_norm[fields_dat],
        thresh_correction = None,
    )

    out = (df_iea, df_iea_norm, )
    
    return out



def get_dict_fuel_to_pp_tech(
    models: 'SISEPUEDEModels'
) -> Dict[str, str]:
    """Map fuels to power generation technologies
    """

    dict_fuel_to_tech = (
        models
        .model_enerprod
        .get_tech_info_dict()
        .get("dict_fuel_to_pp_tech")
    )

    # should revisit this and derive from attribute tables
    dict_fuel_to_tech = dict(
        (k, [x for x in v if "ccs" not in x][0]) 
        for k, v in dict_fuel_to_tech.items()
    )

    return dict_fuel_to_tech



def get_enercons_calib_dfs(
    df_input: pd.DataFrame,
    models: 'SISEPUEDEModels',
    run_enerprod: bool = True,
) -> Tuple[pd.DataFrame]:
    """Get DataFrames projected and used for calibration. Produces three 
        DataFrames:

        (
            df_preliminary,         # full run data frame. If run_enerprod 
                                    # is True, will apply here
            df_preliminary_no_cm,   # run without commercial/municipal
                                    # demand; used to calibrate residential
                                    # energy consumption
            df_preliminary_no_re,   # run without residential demand; used 
                                    # to calibrate commercial/municipal
                                    # energy consumption              
        )
        

    Function Arguments
    ------------------
    df_input : pd.DataFrame
        DataFrame to use as inputs
    models : SISEPUEDEModels
        SISEPUEDEModels object used to retrieve model variables
    
    Keyword Arguments
    -----------------
    run_enerprod : bool
        Run the energy production model? Only applies to df_prelim
    """

    ##  INITIALIZATION AND SHORTCUTS
    
    model = models.model_enercons
    matt = model.model_attributes

    # commercial
    movdvar_sec_comm_app = matt.get_variable(
        model.modvar_scoe_consumpinit_energy_per_mmmgdp_elec, 
    )
    movdvar_sec_comm_heat = matt.get_variable(
        model.modvar_scoe_consumpinit_energy_per_mmmgdp_heat
    )

    # residential
    movdvar_sec_res_app = matt.get_variable(
        model.modvar_scoe_consumpinit_energy_per_hh_elec,
    )
    movdvar_sec_res_heat = matt.get_variable(
        model.modvar_scoe_consumpinit_energy_per_hh_heat,
    )
    

    ##  REMOVE COMM/RES DEMAND AND PROJECT (NO ENERPROD)
    
    df_input_no_cm = df_input.copy()
    df_input_no_cm[movdvar_sec_comm_app.fields] = 0
    df_input_no_cm[movdvar_sec_comm_heat.fields] = 0
    
    # create with no residential demand and run
    df_input_no_re = df_input.copy()
    df_input_no_re[movdvar_sec_res_app.fields] = 0
    df_input_no_re[movdvar_sec_res_heat.fields] = 0
    
    
    # run here--no commercial/municipal
    df_out_preliminary_no_cm = models.project(
        df_input_no_cm,
        include_electricity_in_energy = False,
        verbose = False, 
    )
    
    # run here--no residential
    df_out_preliminary_no_re = models.project(
        df_input_no_re,
        include_electricity_in_energy = False,
        verbose = False,  
    )
    
    
    ##  RUN COMBINED WITH OPTION FOR ENERPROD
    
    df_out_preliminary = models.project(
        df_input,
        include_electricity_in_energy = run_enerprod,
        time_periods_base = range(0, 11),
        time_periods_run = range(0, 11),
        verbose = False, 
    )


    out = (
        df_out_preliminary,
        df_out_preliminary_no_cm,
        df_out_preliminary_no_re,
    )

    return out



def get_iea_dict(
    df_iea_cw: pd.DataFrame,
    model_attributes: 'ModelAttributes',
    delim: str = "|",
    dict_type: str = "iea_fuels",
    field_iea_fuels: str = _FIELD_CW_IEA_FUELS,
    field_iea_fuel_groups: str = _FIELD_CW_IEA_FUEL_GROUPS,
) -> dict:
    """get the dictionary mapping IEA fuels to SSP fuels

    Function Arguments
    ------------------
    df_iea_cw : pd.DataFrame
        DataFrame with crosswalk of information from IEA fuels to SSP 
        fuels
    model_attributes : ModelAttributes
        ModelAttributes object used for variable management
        
    Keyword Arguments
    -----------------
    delim : str
        Delimiter used to unwrap the crosswalk DataFrame
    dict_type : str
        "iea_fuels"
        "iea_fuel_groups"
    field_iea_fuels : str
        Field storing IEA Fuels
    field_iea_fuel_groups : str
        Field storing IEA Fuel Groups
    """

    cat_enfu = model_attributes.get_subsector_attribute(
        model_attributes.subsec_name_enfu, 
        "primary_category"
    )

    field_piv = (
        field_iea_fuels
        if dict_type == "iea_fuels"
        else field_iea_fuel_groups
    )
    
    df_cw = sf.unwrap_df_from_delimiter(
        df_iea_cw
        .dropna(subset = [field_piv], ),
        field_piv,
    )


    if dict_type == "iea_fuels":
        # get output dict
        dict_out = sf.build_dict(df_cw[[field_iea_fuels, cat_enfu]], )
        dict_out = dict(
            (k, ma.clean_schema(v)) 
            for k, v in dict_out.items()
        )

    else:
        dict_out = sf.group_df_as_dict(
            df_cw[[field_piv, cat_enfu]],
            [field_piv],
            fields_out_set = cat_enfu,
        )

        dict_out = dict(
            (k, sorted([ma.clean_schema(x) for x in v]))
            for k, v in dict_out.items()
        )
        

    out = (df_cw, dict_out, )

    return out



def get_iea_df_for_ssp(
    df_iea: pd.DataFrame,
    dict_iea_fuels: Dict[str, str],
    time_periods: 'TimePeriods',
    models: 'SISEPUEDEModels',
    field_iea_fuel: str,
    convert_fuel_to_tech: bool = True,
    dict_category_replacements: Union[Dict[str, str], None] = None,
    field_cw_iea_fuels: str = _FIELD_CW_IEA_FUELS,
    field_cw_iea_fuel_groups: str = _FIELD_CW_IEA_FUEL_GROUPS,
    field_iea_units: str = _FIELD_IEA_UNITS,
    field_iea_value: str = _FIELD_IEA_VALUE,
    field_iea_year: str = _FIELD_IEA_YEAR,
    field_total_base: str = _FIELD_PREFIX_TOTAL,
    fuels_iea_keep: Union[List[str], None] = None,
    modvar: Union['ModelVariable', None] = None,
    normalize: bool = False,
) -> pd.DataFrame:
    """Get the DataFrame of IEA elements in terms of SSP fuels. Option
        to keep fuels/fuel groups not included in dict_iea_fuels

    Function Arguments
    ------------------
    df_iea : pd.DataFrame

    field_iea_fuel : str
        Field in df_iea storing fuels
        
    Keyword Arguments
    -----------------
    dict_category_replacements : Union[Dict[str, str], None]
        Category replacements to map  
    fuels_iea_keep: Union[List[str], None] = None,
    modvar: Union['ModelVariable', None] = None,
    """

    ##  INITIALIZATION

    # check replacements
    if (not isinstance(dict_category_replacements, dict)) and (modvar is None):
        msg = "dict_category_replacements and modvar cannot both be undefined."
        raise TypeError(msg)

    # shortcuts
    matt = models.model_attributes

    # get the data and check the fuel field
    if field_iea_fuel not in df_iea.columns:
        raise KeyError(f"Fuel field '{field_iea_fuel}' not found in input DataFrame")

    # check the units
    units_iea = df_iea[_FIELD_IEA_UNITS].unique()
    if len(units_iea) > 1:
        raise RuntimeError(f"Multiple units found in IEA data. Check the files and write a handler.")
        
    units_iea = units_iea[0].strip().lower()


    ##  START BY AGGREGATING BY SSP FUEL AND REFORMATING
    
    # aggregate
    df_iea[field_iea_fuel] = df_iea[field_iea_fuel].replace(dict_iea_fuels, )
    df_iea = (
        df_iea
        .drop(columns = [field_iea_units])
        .groupby([field_iea_year, field_iea_fuel])
        .sum()
        .reset_index()
    )
    
    # reformat
    df_iea = (
        sf.pivot_df_clean(
            df_iea,
            [field_iea_fuel],
            [field_iea_value],
        )
        .fillna(0.0)
        .rename(columns = {field_iea_year: time_periods.field_year, }, )
    )


    ##  CLEAN AND FILTER
    
    # get the data fields, which will be normalized
    fuels_iea_keep = [] if not sf.islistlike(fuels_iea_keep) else fuels_iea_keep
    fields_dat = [x for x in df_iea.columns if x not in [time_periods.field_year]]
    fields_dat_cat = [x for x in fields_dat if (x in dict_iea_fuels.values()) or (x in fuels_iea_keep)]

    # get total production
    field_total = f"{field_total_base}_{units_iea}"
    df_total = df_iea.copy()
    df_total[field_total] = df_total[fields_dat].sum(axis = 1)
    df_total = df_total.drop(columns = fields_dat, )

    # now drop non-relevant fields from the df out for MSP
    df_iea = df_iea[[time_periods.field_year] + fields_dat_cat]
    if normalize:
        df_iea[fields_dat_cat] = sf.check_row_sums(
            df_iea[fields_dat_cat].to_numpy(),
            thresh_correction = None,
        )
    

    ##  RENAME AND PERFORM ANY FUEL TO TECH CONVERSIONS (FOR POWER PLANTS)
    
    # rename
    dict_category_replacements = (
        matt.get_category_replacement_field_dict(modvar, )
        if modvar is not None
        else dict_category_replacements
    )

    if convert_fuel_to_tech:
        dict_fuel_to_tech = get_dict_fuel_to_pp_tech(models, )

        # get map of fuel to field
        dict_fuel_to_field = dict(
            (k, dict_category_replacements.get(v)) for k, v in dict_fuel_to_tech.items()
        )

    else:
        dict_fuel_to_field = dict_category_replacements
    
    df_iea = df_iea.rename(columns = dict_fuel_to_field, )

    # extract if available
    if modvar is not None:
        df_iea = modvar.get_from_dataframe(
            df_iea,
            extraction_logic = "any_fill",
            fields_additional = [time_periods.field_year], 
        )
        

    # finally, adjust to time periods
    df_iea = (
        time_periods.years_to_tps(
            df_iea
        )
        .drop(columns = [time_periods.field_year])
    )

    
    # return information
    out = (
        df_iea,
        df_total,
        field_total,
        units_iea,
    )
    
    return out



