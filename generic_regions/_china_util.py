import numpy as np
import pandas as pd
import pathlib
from typing import *


##  SOME GLOBALS

# fields
_FIELD_CW_IEA_FUELS = "iea_fuels"
_FIELD_CW_IEA_FUEL_GROUPS = "iea_fuel_groups"
_FIELD_IEA_FUEL = "electricity generation sources in China"
_FIELD_IEA_UNITS = "Units"
_FIELD_IEA_VALUE = "Value"
_FIELD_IEA_YEAR = "Year"






##  FUNCTIONS

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
    



def get_iea(
    path_iea: pathlib.Path,
    dict_iea_fuels: Dict[str, str],
    modvar_msp: 'ModelVariable',
    time_periods: 'TimePeriods',
    models: 'SISEPUEDEModels',
    field_cw_iea_fuels: str = _FIELD_CW_IEA_FUELS,
    field_cw_iea_fuel_groups: str = _FIELD_CW_IEA_FUEL_GROUPS,
    field_iea_fuel: str = _FIELD_IEA_FUEL,
    field_iea_units: str = _FIELD_IEA_UNITS,
    field_iea_value: str = _FIELD_IEA_VALUE,
    field_iea_year: str = _FIELD_IEA_YEAR,
    field_total_base: str = "total",
) -> dict:
    """get the dictionary mapping IEA fuels to SSP fuels

    Function Arguments
    ------------------

    Keyword Arguments
    -----------------
    """

    matt = models.model_attributes
    
    # get the data
    df_elec = pd.read_csv(path_iea, )
    units_iea = df_elec[_FIELD_IEA_UNITS].unique()
    if len(units_iea) > 1:
        raise RuntimeError(f"Multiple units found in electricity data at '{path_iea}'. Check the files and write a handler.")
        
    units_iea = units_iea[0].strip().lower()

    # aggregate
    df_elec[field_iea_fuel] = df_elec[field_iea_fuel].replace(dict_iea_fuels, )
    df_elec = (
        df_elec
        .drop(columns = [field_iea_units])
        .groupby([field_iea_year, field_iea_fuel])
        .sum()
        .reset_index()
    )
    
    # reformat
    df_elec = (
        sf.pivot_df_clean(
            df_elec,
            [field_iea_fuel],
            [field_iea_value],
        )
        .fillna(0.0)
        .rename(columns = {field_iea_year: time_periods.field_year, }, )
    )

    # get the data fields, which will be normalized
    fields_dat = [x for x in df_elec.columns if x not in [time_periods.field_year]]
    fields_dat_cat = [x for x in fields_dat if x in dict_iea_fuels.values()]

    # get total production
    field_total = f"{field_total_base}_{units_iea}"
    df_total = df_elec.copy()
    df_total[field_total] = df_total[fields_dat].sum(axis = 1)
    df_total = df_total.drop(columns = fields_dat, )

    # now drop non-relevant fields from the df out for MSP
    df_elec = df_elec[[time_periods.field_year] + fields_dat_cat]
    df_elec[fields_dat_cat] = sf.check_row_sums(
        df_elec[fields_dat_cat].to_numpy(),
        thresh_correction = None,
    )

    # rename
    dict_msp = matt.get_category_replacement_field_dict(modvar_msp, )
    dict_fuel_to_tech = get_dict_fuel_to_pp_tech(models, )
    dict_fuel_to_field = dict(
        (k, dict_msp.get(v)) for k, v in dict_fuel_to_tech.items()
    )

    df_elec = df_elec.rename(columns = dict_fuel_to_field, )
    df_elec = modvar_msp.get_from_dataframe(
        df_elec,
        extraction_logic = "any_fill",
        fields_additional = [time_periods.field_year], 
    )

    # finally, adjust to time periods
    df_elec = (
        time_periods.years_to_tps(
            df_elec
        )
        .drop(columns = [time_periods.field_year])
    )

    
    
    # return information
    out = (
        df_elec,
        df_total,
        field_total,
        units_iea,
    )
    
    return out



def get_iea_dict(
    path_iea_cats: pathlib.Path,
    model_attributes: 'ModelAttributes',
    delim: str = "|",
    field_iea_fuels: str = _FIELD_CW_IEA_FUELS,
    field_iea_fuel_groups: str = _FIELD_CW_IEA_FUEL_GROUPS,
) -> dict:
    """get the dictionary mapping IEA fuels to SSP fuels

    Function Arguments
    ------------------

    Keyword Arguments
    -----------------
    """

    cat_lndu = model_attributes.get_subsector_attribute(
        model_attributes.subsec_name_enfu, 
        "primary_category"
    )

    
    df_cw = sf.unwrap_df_from_delimiter(
        pd.read_csv(path_iea_cats, )
        .dropna(
            subset = [field_iea_fuels],
        ),
        field_iea_fuels,
    )


    # get output dict
    dict_out = sf.build_dict(df_cw[[field_iea_fuels, cat_lndu]], )
    dict_out = dict(
        (k, ma.clean_schema(v)) 
        for k, v in dict_out.items()
    )

    out = (df_cw, dict_out, )

    return out
    


def overwrite_msp_from_iea(
    path_iea: pathlib.Path,
    path_iea_cats: pathlib.Path,
    modvar_msp: 'ModelVariable',
    time_periods: 'TimePeriods',
    models: 'SISEPUEDEModels',
) -> pd.DataFrame:
    """
    path_iea : pathlib.Path
        Path to IEA data to read
    path_iea_cats : pathlib.Path
        Path to IEA crosswalk to read
    time_periods : TimePeriods
        Time periods management
    models : SISEPUEDEModels
        SISEPUEDEModels object used for variable/dict access
    """

    # get the crosswalk
    df_cw, dict_iea_fuels = get_iea_dict(path_iea_cats, models.model_attributes, )

    #
    (
        df_elec,
        df_total,
        field_total,
        units_iea,
    ) = get_iea(
        path_iea,
        dict_iea_fuels,
        modvar_msp,
        time_periods,
        models,
    )




    #
    #    LEGACY CODE APPROACH
    #
    """
    if False:
        df_elec2 = (
            pd.merge(
                df_input[[time_periods.field_time_period]],
                df_elec,
                how = "left",
            )
            .interpolate()
            .ffill()
            .bfill()
        )
    
    
    
    
    df_elec2 = df_elec.copy()
    
    t_next = df_elec2[time_periods.field_time_period].max() + 1
    
    while t_next - 1 <= max(time_periods.all_time_periods):
        df_elec2 = putil.repeat_value(
            df_elec2, 
            [t_next],
            15,
            "linear_regression",
            time_periods.field_time_period,
            _MODVAR_MSP.fields,
        )
    
        t_next += 1
    """

    # regress to next value, then repeat
    t_max = df_elec[time_periods.field_time_period].max()
    df_elec = putil.repeat_value(
        df_elec, 
        [x for x in time_periods.all_time_periods if x > t_max],
        10,
        "linear_regression",
        time_periods.field_time_period,
        modvar_msp.fields,
    )
    
    # filter out unneeded time periods
    df_elec = (
        df_elec[
            df_elec[time_periods.field_time_period]
            .isin(time_periods.all_time_periods, )
        ]
        .reset_index(drop = True, )
    )
    
    
    # overwrite
    df_new = sf.match_df_to_target_df(
        df_input,
        df_elec,
        [time_periods.field_time_period],
        try_interpolate = True, 
    )


    return df_new