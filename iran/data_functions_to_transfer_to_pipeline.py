#
#   FUNCTIONS THAT WILL EVENTUALLY BY PORTED TO PIPELINE
#
from typing import *
import lib.data_objects as do
import sisepuede.core.model_attributes as ma
import sisepuede.core.support_classes as sc


#
#    DEFINE FUNCTIONS FOR SINGLE REGION, CAN BE EXPANDED TO MULTIPLE REGIONS
#

def aggregate_and_pivot_iea_fuel_single_region(
    dict_files: dict,
    key_dict: str,# = "electricity generation sources",
    modvar: Union[str, 'ModelVariable', None],# = "NemoMod MinShareProduction",
    model_attributes: ma.ModelAttributes,
    time_periods: sc.TimePeriods,
    regions: sc.Regions,
    calculate_proportion: bool = False,
    cats_keep: Union[List[str], None] = None,
    convert_units: bool = True,
    dataset_iea: do.DatasetIEA = iea_info,
    dict_cat_repl: Union[dict, None] = None,
    field_tech: str = FIELD_IEA_TECH_TMP,
) -> pd.DataFrame:
    """
    Build the minimum share of production data from country profile data


    Function Arguments
    ------------------
    - dict_files: dictionary containing IEA files
    - key_dict: key in the dictionary; also assumed to be a component of
        the field storing fuels
    - modvar: name of variable to assign or ModelVariable object. If None, 
        then unit conversions cannot be performed. 
    
    
    Keyword Arguments
    -----------------
    - calculate_proportion: calculate the value as a proportion?
    - cats_keep: optional specification of categories (POST replacement) to keep
    - convert_units: if not calculating as a proporition, try to convert
        units?
    - dataset_iea: DatasetIEA object used for field access etc.
    - dict_cat_repl: optional dictionary used to replace fuel categories
        with others
    - field_tech: field used to generate technologies
    """

    ##  SOME INIT
    
    df = (
        dict_files
        .get(key_dict)
        .copy()
    )
    field_fuel_to_replace = [x for x in df.columns if (key_dict in x)]
    if len(field_fuel_to_replace) == 0:
        return None

    
    (
        df
        .rename(
            columns = {
                field_fuel_to_replace[0]: field_tech,
            },
            inplace = True,
        )
    )

    # get the model variable
    modvar = model_attributes.get_variable(modvar, )
    

    ##  GROUP BY YEAR AND BUILD SHARES

    
    if not calculate_proportion:
        df_out = df.copy()

        if (dataset_iea.field_units in df.columns) and (modvar is not None):

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
            df_out[field_tech].replace(dict_cat_repl, inplace = True, )
        
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
            
        df_out[field_tech].replace(dict_cats_to_field, inplace = True, )

    
    # aggregate by tech/year
    df_out = (
        df_out
        .fillna(0.0)
        .groupby([field_tech, dataset_iea.field_year])
        .sum()
        .reset_index(drop = False)
    )

    ##  REFORMATTING TO WIDE BY FIELD

    df_out = sf.pivot_df_clean(
        df_out,
        [field_tech],
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

    df_out = (
        pd.merge(
            time_periods.get_time_period_df(),
            df_out,
            how = "left",
        )
        .interpolate(method = "bfill")
        .interpolate(method = "ffill")
    )

    
    return df_out