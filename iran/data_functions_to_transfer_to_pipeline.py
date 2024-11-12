#
#   FUNCTIONS THAT WILL EVENTUALLY BY PORTED TO PIPELINE
#
from typing import *
import lib.data_objects as do
import lib.process_utilities as pu
import lib.sisepuede_data_constructs as dc
import numpy as np
import os, os.path
import pandas as pd
import re
import sisepuede.core.model_attributes as ma
import sisepuede.core.support_classes as sc
import sisepuede.utilities._toolbox as sf
import warnings


#
#    DEFINE FUNCTIONS FOR SINGLE REGION, CAN BE EXPANDED TO MULTIPLE REGIONS
#


def aggregate_and_pivot_iea_fuel(
    df: pd.DataFrame,
    field_index_iea: str,
    modvar: Union[str, ma.mv.ModelVariable, None],
    constructs: dc.SISEPUEDEDataConstructs,
    dataset_iea: do.DatasetIEA, # = iea_info,
    calculate_proportion: bool = False,
    cats_keep: Union[List[str], None] = None,
    convert_units: bool = True,
    dict_cat_repl: Union[dict, None] = None,
    field_region: Union[str, None] = None,
    time_periods_type: str = "historical",
) -> pd.DataFrame:
    """
    Build the minimum share of production data from country profile data

    Function Arguments
    ------------------
    - df: data frame containing IEA data to aggregate and pivot
    - field_index_iea: field in the IEA data frame used to index the variable. 
        Depends on the dataset being called
    - modvar: name of variable to assign or ModelVariable object. If None, 
        then unit conversions cannot be performed.
    - constructs: SISEPUEDEDataConstructs to use for access to SISEPUEDE
        coordination structures
    - dataset_iea: DatasetIEA object used for field access etc.
    
    
    Keyword Arguments
    -----------------
    - calculate_proportion: calculate the value as a proportion?
    - cats_keep: optional specification of categories (POST replacement) to keep
    - convert_units: if not calculating as a proporition, try to convert
        units?
    - dict_cat_repl: optional dictionary used to replace fuel categories with 
        others
    - field_region: optional specification of field region to group on
    - time_periods_type: time periods to keep. 
        * "historical": keeps only those in the historical data
        * "all": maps a value forward for all
    """

    field_region = (
        constructs.regions.field_iso
        if not isinstance(field_region, str)
        else field_region
    )

    if field_region not in df.columns:
        return df


    dfg = df.groupby([field_region])
    df_out = []
    
    for (region, df_cur) in dfg:
        
        df_cur = aggregate_and_pivot_iea_fuel_single_region(
            df_cur,
            field_index_iea,
            modvar,
            constructs,
            dataset_iea, # = iea_info,
            calculate_proportion = calculate_proportion,
            cats_keep = cats_keep,
            convert_units = convert_units,
            dict_cat_repl = dict_cat_repl,
            time_periods_type = time_periods_type,
        )

        df_cur[field_region] = region[0]
        df_out.append(df_cur)
    
    # get fields to order and concatenate
    fields_ind = [field_region, constructs.time_periods.field_time_period]
    fields_ord = [x for x in df_out[0].columns if x not in fields_ind]

    df_out = (
        pd.concat(
            df_out,
            axis = 0,
        )
        .get(fields_ind + fields_ord)
        .reset_index(drop = True, )
    )

    return df_out

     

def aggregate_and_pivot_iea_fuel_single_region(
    df: pd.DataFrame,
    field_index_iea: str,
    modvar: Union[str, ma.mv.ModelVariable, None],
    constructs: dc.SISEPUEDEDataConstructs,
    dataset_iea: do.DatasetIEA, # = iea_info,
    calculate_proportion: bool = False,
    cats_keep: Union[List[str], None] = None,
    convert_units: bool = True,
    dict_cat_repl: Union[dict, None] = None,
    time_periods_type: str = "historical",
) -> pd.DataFrame:
    """
    Build the minimum share of production data from country profile data


    Function Arguments
    ------------------
    - df: data frame containing IEA data to aggregate and pivot
    - field_index_iea: field in the IEA data frame used to index the variable. 
        Depends on the dataset being called
    - modvar: name of variable to assign or ModelVariable object. If None, 
        then unit conversions cannot be performed.
    - constructs: SISEPUEDEDataConstructs to use for access to SISEPUEDE
        coordination structures
    - dataset_iea: DatasetIEA object used for field access etc.
    
    
    Keyword Arguments
    -----------------
    - calculate_proportion: calculate the value as a proportion?
    - cats_keep: optional specification of categories (POST replacement) to keep
    - convert_units: if not calculating as a proporition, try to convert
        units?
    - dict_cat_repl: optional dictionary used to replace fuel categories with 
        others
    - time_periods_type: time periods to keep. 
        * "historical": keeps only those in the historical data
        * "all": maps a value forward for all
    """

    ##  SOME INIT

    if field_index_iea not in df.columns:
        return df


    # set some shortcuts
    model_attributes = constructs.sisepuede_file_struct.model_attributes
    regions = constructs.regions
    time_periods = constructs.time_periods

    # get the model variable
    modvar = model_attributes.get_variable(modvar, )
    

    ##  GROUP BY YEAR AND BUILD SHARES

    
    if not calculate_proportion:
        df_out = df.copy()
        
        try_units_conversion = convert_units 
        try_units_conversion &= (dataset_iea.field_units in df.columns) 
        try_units_conversion &= (modvar is not None)

        if try_units_conversion:

            # get the source units (in IEA)
            units_source = df[dataset_iea.field_units].unique()
            if len(units_source) > 1:
                raise RuntimeError(f"Check units in {key_dict}: multiple units found. Cannot convert uniformly.")

            units_source = str(units_source[0]).lower().strip()

            # get target units and scalar to convert source to target
            units_target = modvar.attribute("unit_energy")
            unit_manager = model_attributes.get_unit("energy")
            scalar = unit_manager.convert(
                units_source,
                units_target
            )

            df_out[dataset_iea.field_value] = df_out[dataset_iea.field_value].to_numpy() * scalar

                
        
    else:
        df_out = []
    
        dfg = df.groupby([dataset_iea.field_year])
        for yr, df in dfg:
            yr = yr[0]
    
            df[dataset_iea.field_value] = df[dataset_iea.field_value]/df[dataset_iea.field_value].sum()
            df.drop(
                columns = [
                    dataset_iea.field_units,
                ],
                inplace = True,
            )
    
            df_out.append(df)

        df_out = pd.concat(df_out, )
        
    
    ##  CONCATENATE, CLEAN CATEGORIES, AND AGGREGATE

    # do preliminary replacement?
    if isinstance(dict_cat_repl, dict):
            df_out[field_index_iea].replace(dict_cat_repl, inplace = True, )
        
    if modvar is not None:
        subsec = model_attributes.get_variable_subsector(modvar, )
        attr_subsec = model_attributes.get_attribute_table(subsec, )
        dict_cats_to_field = model_attributes.get_category_replacement_field_dict(modvar, )
    
        # categories to keep
        cats_keep = (
            [x for x in attr_subsec.key_values if x in cats_keep]
            if sf.islistlike(cats_keep)
            else attr_subsec.key_values
        )
        # fields to keep
        fields_universe = [dict_cats_to_field.get(x) for x in cats_keep if x in dict_cats_to_field.keys()] 
            
        df_out[field_index_iea].replace(dict_cats_to_field, inplace = True, )

    
    # aggregate by tech/year
    df_out = (
        df_out
        .fillna(0.0)
        .groupby([field_index_iea, dataset_iea.field_year])
        .sum()
        .reset_index(drop = False)
    )

    ##  REFORMATTING TO WIDE BY FIELD

    df_out = sf.pivot_df_clean(
        df_out,
        [field_index_iea],
        [dataset_iea.field_value]
    )

    df_out = (
        time_periods
        .years_to_tps(
            df_out,
            field_year = dataset_iea.field_year,
        )
        .drop(columns = [dataset_iea.field_year])
    )

    
    # get as a variable and use "any_fill" method to pull all--then, filter on power plants only
    fields_additional = [
        x for x in model_attributes.sort_ordered_dimensions_of_analysis
        if x in df_out.columns
    ]


    if modvar is not None:
        df_out = (
            modvar
            .get_from_dataframe(
                df_out,
                extraction_logic = "any_fill",
                fields_additional = fields_additional,
                fill_value = 0.0,
            )
            .get(fields_additional + fields_universe)
        )

   


    ##  FINALLY, MERGE TO ALL TIME PERIODS AND INTERPOLATE WHERE NEEDED

    merge_type = "left" if time_periods_type == "all" else "inner"

    df_out = (
        pd.merge(
            time_periods.get_time_period_df(),
            df_out,
            how = merge_type,
        )
        .interpolate(method = "bfill")
        .interpolate(method = "ffill")
    )

    
    return df_out



def combine_iea_csvs_to_table(
    reader: pu.InputsReader,
    constructs: dc.SISEPUEDEDataConstructs,
    input_dataset: str,
    default_field_renamed: str = "iea_fuel",
    fillval: Union[float, int, None] = None,
    key_config_field_index_rename: str = "field_iea_index_renamed",
) -> Union[pd.DataFrame, None]:
    """
    Using a reader, combine IEA individiual CSVs

    Function Arguments
    ------------------
    - reader: InputsReader storing input data
    - constructs: SISEPUEDE constructs with access to regions, 
        model_attributes, etc.
    - input_dataset: input dataset (first-level key under reader.inputs) 
        to retrieve
        
    Keyword Arguments
    -----------------
    - default_field_renamed: default field name to use for renaming
        IEA fields if not specified in the configuration. In each of the 
        datasets, a key variable indexing field is named with 
        country-specific information. This needs to be renamed to something 
        generic to allow for concatenation.
    - fillval: optional value to use for filling values
    - key_config_field_index_rename: key in the configuration below the 
        dataset storing the new index field name
    """

    # get regular expressions
    (regex_csv, regex_field) = get_regular_expressions_for_iea_csvs(
        reader,
        f"{reader.key_inputs}.{input_dataset}"
    )

    # check path--if not available 
    fp_read = reader.dict_paths_full.get(input_dataset)
    if fp_read is None:
        return None


    # otherwise, get available files and get the new index field name
    files = [
        x for x in os.listdir(fp_read) 
        if regex_csv.match(x) is not None
    ]    
    
    files.sort() # sort so that we don't have to 

    
    field_index_new = (
        reader
        .config
        .get(
            f"{reader.key_inputs}.{input_dataset}.{key_config_field_index_rename}",
            return_on_none = default_field_renamed,
        )
    )


    ##  ITERATE OVER FILES TO BUILD
    
    df_out = []

    for fn in files:
        # set path to read
        fp_file = fp_read.joinpath(fn)

        ##  CHECK THE REGION
        
        region_base = regex_csv.match(fn).groups()[0]
        region = (
            constructs
            .regions
            .data_func_try_isos_from_countries(
                region_base, 
                missing_iso_flag = "MISSING",
            )
        )

        # region not found
        if region[0] == "MISSING":
            warnings.warn(
                f"""WARNING: region '{region_base}' not found. Check the 
                name and ppotentially update the IEA regions dictionary ar
                support_classes.Regions `dict_iea_countries_lc_to_regions`.
                """
            )
            continue

        region = region[0]

        

        # try reading the file in
        try:
            df = pd.read_csv(fp_file)
        except:
            continue

        # next, let's get the column we need to rename
        field_rename = [x for x in df.columns if regex_field.match(x) is not None]
        if len(field_rename) > 1:
            warnings.warn(
                f"WARNING: multiple IEA index fields fund at path '{fp_file}'. Check the data before trying to read in."
            )
            continue

        if len(field_rename) == 1:
            df.rename(
                columns = {field_rename[0]: field_index_new},
                inplace = True,
            )

        # add region and set output order
        field_region = constructs.regions.field_iso
        df[field_region] = region
        
        fields_ord = [x for x in df.columns if x not in [field_index_new, field_region]]
        df = df[[field_region, field_index_new] + fields_ord]

        df_out.append(df)


    # 
    df_out = (
        pd.concat(df_out, axis = 0, )
        if len(df_out) > 0
        else None
    )

    if sf.isnumber(fillval):
        df_out.fillna(fillval, inplace = True)

    df_out = (
        df_out
        .dropna()
        .reset_index(drop = True, )
    )

    return df_out



def get_dict_enfu_cat_to_scoe_vars(
    model_enercons: 'EnergyConsumption',
) -> dict:
    """
    Map enfu categories to fuel fraction variables. Builds the following
        dictionaries

        * dict_cat_enfu_to_scoe_heat_fuel_fraction
        * dict_cat_enfu_to_scoe_heat_fuel_efficiency
    """
    # some initialization
    dict_fuel_frac_to_eff = model_enercons.modvar_dict_scoe_fuel_fractions_to_efficiency_factors
    matt = model_enercons.model_attributes
    
    pycatdisp_enfu = matt.get_subsector_attribute(
        matt.subsec_name_enfu, 
        "pycategory_primary_element",
    )


    ##  BUILD DICTS

    # init
    dict_cat_enfu_to_scoe_heat_fuel_fraction = {}
    dict_cat_enfu_to_scoe_heat_fuel_efficiency = {}
    
    for k, v in dict_fuel_frac_to_eff.items():

        modvar_frac = matt.get_variable(k)
        modvar_eff = matt.get_variable(v)

        key_new = modvar_frac.get_property(pycatdisp_enfu)
        key_new = ma.clean_schema(key_new)

        dict_cat_enfu_to_scoe_heat_fuel_fraction.update({key_new: modvar_frac})
        dict_cat_enfu_to_scoe_heat_fuel_efficiency.update({key_new: modvar_eff})

    
    out = (
        dict_cat_enfu_to_scoe_heat_fuel_fraction,
        dict_cat_enfu_to_scoe_heat_fuel_efficiency,
    )

    return out



def get_regular_expressions_for_iea_csvs(
    reader: pu.InputsReader,
    key_var: str,
    key_regex_csv: str = "regex_csv",
    key_regex_field_rename: str = "regex_field_rename",
) -> Tuple:
    """
    Retrieve regular expressions for file matching and associated field
        for the variable specified in key (input.VARIABLENAME)
    """

    # try CSVs
    regex_csv = reader.config.get(f"{key_var}.{key_regex_csv}")
    if regex_csv is not None:
        regex_csv = re.compile(regex_csv)

    # try field for fuels
    regex_field = reader.config.get(f"{key_var}.{key_regex_field_rename}")
    if regex_field is not None:
        regex_field = re.compile(regex_field)
    
    out = (regex_csv, regex_field)

    return out



def get_scalar_for_iea_conversion(
    df_consumption: pd.DataFrame,
    modvar: Union[str, ma.mv.ModelVariable],
    model_attributes: ma.ModelAttributes,
    dataset_iea: do.DatasetIEA,
    unit_type: str = "energy",
) -> float:
    """
    Get a scalar to convert units in an SCOE table into target variable units
        (useful when not converting to fields because categories don't align)
    """
    units_source = df_consumption[dataset_iea.field_units].unique()
    units_source = str(units_source[0]).lower().strip()
    
    # get target units and scalar to convert source to target
    units_target = modvar.attribute(f"unit_{unit_type}")
    unit_manager = model_attributes.get_unit(unit_type)
    scalar = unit_manager.convert(
        units_source,
        units_target
    )

    return scalar



def get_wide_iea_data_in_variable_units(
    reader: pu.InputsReader,
    constructs: dc.SISEPUEDEDataConstructs,
    input_dataset: str,
    modvar: Union[str, ma.mv.ModelVariable],
    dataset_iea: do.DatasetIEA,
    default_field_renamed: str = "iea_fuel",
    fillval: Union[float, int, None] = 0.0,
    key_config_field_index_rename: str = "field_iea_index_renamed",
) -> Union[pd.DataFrame, None]:
    """
    Using a reader, retrieve an IEA dataset in terms of some target
        units. Does not convert to SISEPUEDE fields.

    Function Arguments
    ------------------
    - reader: InputsReader storing input data
    - constructs: SISEPUEDE constructs with access to regions, 
        model_attributes, etc.
    - input_dataset: input dataset (first-level key under reader.inputs) 
        to retrieve
    - modvar: ModelVariable to use for units target
    - dataset_iea: DatasetIEA to use for access to fields
        
    Keyword Arguments
    -----------------
    - default_field_renamed: default field name to use for renaming
        IEA fields if not specified in the configuration. In each of the 
        datasets, a key variable indexing field is named with 
        country-specific information. This needs to be renamed to something 
        generic to allow for concatenation.
    - fillval: optional value to use for filling values
    - key_config_field_index_rename: key in the configuration below the 
        dataset storing the new index field name
    """

    df_consumption_0 = combine_iea_csvs_to_table(
        reader,
        constructs,
        input_dataset,
        default_field_renamed = default_field_renamed,
        fillval = fillval,
        key_config_field_index_rename = key_config_field_index_rename,
    )  

    field_index = reader.get_input(f"{input_dataset}.{key_config_field_index_rename}")
    df_consumption = aggregate_and_pivot_iea_fuel(
        df_consumption_0,
        field_index,
        None,
        constructs,
        dataset_iea,
        convert_units = False,
    ) 

    # convert units
    scalar = get_scalar_for_iea_conversion(
        df_consumption_0, 
        modvar,
        constructs.sisepuede_file_struct.model_attributes,
        dataset_iea,
    )

    fields_iea = list(df_consumption_0[field_index].unique())
    df_consumption[fields_iea] = df_consumption[fields_iea].to_numpy() * scalar
    
    df_consumption = (
        df_consumption
        .fillna(fillval, )
        .drop(columns = [dataset_iea.field_units], )
    )


    return df_consumption



def select_iea_groups_by_fuel(
    df: pd.DataFrame,
    df_cw: pd.DataFrame,
    model_attributes: ma.ModelAttributes,
    delim: str = "|",
) -> dict:
    """
    For each fuel, select the group in the DataFrame to map it to. 
    """

    prim_category_enfu = model_attributes.get_subsector_attribute(
        model_attributes.subsec_name_enfu, 
        "primary_category",
    )

    fields_search = [x for x in df_cw.columns if x != prim_category_enfu]
    df_cw_cur = (
        df_cw
        .dropna(
            subset = fields_search,
            how = "all",
        )
    )
    

    dict_fuel_to_field = {}
    
    # iterate through rows; fields are ordered
    for i, row in df_cw_cur.iterrows():

        fuel = ma.clean_schema(row[prim_category_enfu])
        iea_class = None
        
        for field in fields_search:
            
            if iea_class is not None:
                continue
                
            # get potential classes
            classes_cur = str(row[field]).split(delim)
            for class_cur in classes_cur:
                if (class_cur in df.columns) and (iea_class is None):
                    iea_class = class_cur


        if iea_class is not None:
            dict_fuel_to_field.update({fuel: iea_class})
    
    return dict_fuel_to_field