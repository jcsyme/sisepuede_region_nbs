"""Build groups for uncertinty exploration in templates
"""
import numpy as np
import pandas as pd
import sisepuede.core.support_classes as sc
import sisepuede.utilities._toolbox as sf
from typing import *




########################
#    CALL FUNCTIONS    #
########################

def _build_uncertainty_from_variable(
    modvar: Union[str, 'ModelVariable', List[Union[str, 'ModelVariable']]],
    scale_range: Tuple[float, float],
    trajgroup: int,
    model_attributes: 'ModelAttributes',
    field_max: str,
    field_min: str,
    field_trajgroup: str,
    field_variable: str,
    categories: Union[str, List[str], None] = None,
    each_modvar_gets_unique_trajgroup: bool = False,
    transformers: Union['Transformers', None] = None,
    **kwargs,
) -> pd.DataFrame:
    """Build uncertainties for energy storage

    Function Arguments
    ------------------
    modvar : Union[str, 'ModelVariable', List[Union[str, 'ModelVariable']]]
        ModelVariable to use for 
    scale_range : Tuple[float, float]
        Range of (min, max) to scale
    trajgroup : int
        trajectory group to assign
    field_max : str
        Field used to store max range scalar in templates
    field_min : str
        Field used to store min range scalar in templates
    field_trajgroup : str
        Field storing the trajectory group
    field_variable : str
        Field used to store the variable field name

    Keyword Arguments
    -----------------
    categories : Union[str, List[str], None]
        Optional categories restriction to place
    each_modvar_gets_unique_trajgroup : bool
        Set to True to assign each ModelVariable a unique trajgroup
    """
    
    # variables to tie into groups
    modvar = [modvar] if not sf.islistlike(modvar) else modvar
    modvar = [model_attributes.get_variable(x, ) for x in modvar]
    modvar = [x for x in modvar if x is not None]
    
    if len(modvar) == 0:
        out = (None, trajgroup)
        return out

    # check categories
    categories = [categories] if isinstance(categories, str) else categories
    categories = None if not sf.islistlike(categories) else categories
    df_out = []

    
    for modv in modvar:
        
        # build component dataframe
        fields_combine = (
            modv.fields
            if categories is None
            else modv.build_fields(
                category_restrictions = categories,
            )
        )
        
        df_component = pd.DataFrame(
            {
                field_variable: fields_combine,
                field_max: [scale_range[1] for x in fields_combine],
                field_min: [scale_range[0] for x in fields_combine],
                field_trajgroup: [trajgroup for x in fields_combine],
            }
        )

        df_out.append(df_component, )

        trajgroup += int(each_modvar_gets_unique_trajgroup)

    df_out = sf._concat_df(df_out, )
    #print(f"{len(modvar)} -> {df_out.shape[0]}")
    # move to next iteration and return
    trajgroup += int(not each_modvar_gets_unique_trajgroup)
    out = (df_out, trajgroup, )
    
    return out



def _build_uncertainty_fuel_prices(
    trajgroup: int,
    model_attributes: 'ModelAttributes',
    field_max: str,
    field_min: str,
    field_trajgroup: str,
    field_variable: str,
    **kwargs,
) -> pd.DataFrame:
    """Build uncertainties for technologies. 

    Function Arguments
    ------------------
    trajgroup : int
        trajectory group to assign
    field_max : str
        Field used to store max range scalar in templates
    field_min : str
        Field used to store min range scalar in templates
    field_trajgroup : str
        Field storing the trajectory group
    field_variable : str
        Field used to store the variable field name
    """


    _CATS_ENFU_FF = [
        "fuel_coal",
        "fuel_coke",
        "fuel_crude",
        "fuel_diesel",
        "fuel_gasoline",
        "fuel_hydrocarbon_gas_liquids",
        "fuel_kerosene",
        "fuel_natural_gas",
        "fuel_oil"
    ]
    scalar_range = (0.5, 1.5)


    # variables to tie into groups
    modvar_fuel_price_gravimetric = model_attributes.get_variable("Gravimetric Fuel Price")
    modvar_fuel_price_thermal = model_attributes.get_variable("Thermal Fuel Price")
    modvar_fuel_price_volumetric = model_attributes.get_variable("Volumetric Fuel Price")
    modvars = [
        modvar_fuel_price_gravimetric,
        modvar_fuel_price_thermal,
        modvar_fuel_price_volumetric
    ]

    fields_combine = []
    for mv in modvars:
        fields_combine += mv.build_fields(
            category_restrictions = _CATS_ENFU_FF
        )


    df_component = pd.DataFrame(
        {
            field_variable: fields_combine,
            field_max: [scalar_range[1] for x in fields_combine],
            field_min: [scalar_range[0] for x in fields_combine],
            field_trajgroup: [trajgroup for x in fields_combine],
        }
    )

    # push to next
    trajgroup += 1
    out = (df_component, trajgroup, )

    return out



def _build_uncertainty_fuel_exports(
    trajgroup: int,
    model_attributes: 'ModelAttributes',
    field_max: str,
    field_min: str,
    field_trajgroup: str,
    field_variable: str,
    transformers: Union['Transformers', None] = None,
    **kwargs,
) -> pd.DataFrame:
    """Build uncertainties for fuel exports

    Function Arguments
    ------------------
    trajgroup : int
        trajectory group to assign
    field_max : str
        Field used to store max range scalar in templates
    field_min : str
        Field used to store min range scalar in templates
    field_trajgroup : str
        Field storing the trajectory group
    field_variable : str
        Field used to store the variable field name
    """

    out = _build_uncertainty_from_variable(
        "Fuel Exports",
        (0.75, 1.75),
        trajgroup,
        model_attributes,
        field_max,
        field_min,
        field_trajgroup,
        field_variable,
        **kwargs,
    )
    
    return out



def _build_uncertainty_gdp(
    trajgroup: int,
    model_attributes: 'ModelAttributes',
    field_max: str,
    field_min: str,
    field_trajgroup: str,
    field_variable: str,
    transformers: Union['Transformers', None] = None,
    **kwargs,
) -> pd.DataFrame:
    """Build uncertainties for fuel exports

    Function Arguments
    ------------------
    trajgroup : int
        trajectory group to assign
    field_max : str
        Field used to store max range scalar in templates
    field_min : str
        Field used to store min range scalar in templates
    field_trajgroup : str
        Field storing the trajectory group
    field_variable : str
        Field used to store the variable field name
    """

    out = _build_uncertainty_from_variable(
        "GDP",
        (0.9, 1.1),
        trajgroup,
        model_attributes,
        field_max,
        field_min,
        field_trajgroup,
        field_variable,
        **kwargs,
    )
    
    return out



def _build_uncertainty_inen_demand_scalar(
    trajgroup: int,
    model_attributes: 'ModelAttributes',
    field_max: str,
    field_min: str,
    field_trajgroup: str,
    field_variable: str,
    transformers: Union['Transformers', None] = None,
    **kwargs,
) -> pd.DataFrame:
    """Build uncertainties for energy storage

    Function Arguments
    ------------------
    trajgroup : int
        trajectory group to assign
    field_max : str
        Field used to store max range scalar in templates
    field_min : str
        Field used to store min range scalar in templates
    field_trajgroup : str
        Field storing the trajectory group
    field_variable : str
        Field used to store the variable field name
    """
    

    out = _build_uncertainty_from_variable(
        "Industrial Energy Demand Scalar",
        (0.9, 2.0),
        trajgroup,
        model_attributes,
        field_max,
        field_min,
        field_trajgroup,
        field_variable,
        categories = "other_product_manufacturing",
        each_modvar_gets_unique_trajgroup = True,
        **kwargs,
    )
    
    return out



def _build_uncertainty_inen_elasticities(
    trajgroup: int,
    model_attributes: 'ModelAttributes',
    field_max: str,
    field_min: str,
    field_trajgroup: str,
    field_variable: str,
    transformers: Union['Transformers', None] = None,
    **kwargs,
) -> pd.DataFrame:
    """Build uncertainties for energy storage

    Function Arguments
    ------------------
    trajgroup : int
        trajectory group to assign
    field_max : str
        Field used to store max range scalar in templates
    field_min : str
        Field used to store min range scalar in templates
    field_trajgroup : str
        Field storing the trajectory group
    field_variable : str
        Field used to store the variable field name
    """
    
    modvars = [
        "Elasticity of Industrial Production to GDP",
        "Elasticity of Product Use Rate to GDP per Capita",
    ]

    out = _build_uncertainty_from_variable(
        modvars,
        (0.75, 1.25),
        trajgroup,
        model_attributes,
        field_max,
        field_min,
        field_trajgroup,
        field_variable,
        each_modvar_gets_unique_trajgroup = True,
        **kwargs,
    )


    ##  UPDATE ELASTICITY FOR OTHER PRODUCT MANUFACTURING (data centers primarily)
    cat_ippu = "other_product_manufacturing"
    modvar = model_attributes.get_variable("Elasticity of Industrial Production to GDP")
    field = modvar.build_fields(category_restrictions = cat_ippu, )

    w = np.where(out[0][field_variable].isin([field]))[0]
    print(w)
    if len(w) > 0:
        out[0][field_max].loc[w[0]] = 1.5
        out[0][field_min].loc[w[0]] = 1
    
    return out



def _build_uncertainty_population(
    trajgroup: int,
    model_attributes: 'ModelAttributes',
    field_max: str,
    field_min: str,
    field_trajgroup: str,
    field_variable: str,
    transformers: Union['Transformers', None] = None,
    **kwargs,
) -> pd.DataFrame:
    """Build uncertainties for fuel exports

    Function Arguments
    ------------------
    trajgroup : int
        trajectory group to assign
    field_max : str
        Field used to store max range scalar in templates
    field_min : str
        Field used to store min range scalar in templates
    field_trajgroup : str
        Field storing the trajectory group
    field_variable : str
        Field used to store the variable field name
    """

    out = _build_uncertainty_from_variable(
        "Population",
        (0.925, 1.075),
        trajgroup,
        model_attributes,
        field_max,
        field_min,
        field_trajgroup,
        field_variable,
        **kwargs,
    )
    
    return out



def _build_uncertainty_scoe_elasticities(
    trajgroup: int,
    model_attributes: 'ModelAttributes',
    field_max: str,
    field_min: str,
    field_trajgroup: str,
    field_variable: str,
    transformers: Union['Transformers', None] = None,
    **kwargs,
) -> pd.DataFrame:
    """Build uncertainties for energy storage

    Function Arguments
    ------------------
    trajgroup : int
        trajectory group to assign
    field_max : str
        Field used to store max range scalar in templates
    field_min : str
        Field used to store min range scalar in templates
    field_trajgroup : str
        Field storing the trajectory group
    field_variable : str
        Field used to store the variable field name
    """
    
    modvars = [
        "SCOE Elasticity of Per GDP Electrical Applicance Demand to GDP Per Capita",
        "SCOE Elasticity of Per GDP Heat Energy Demand to GDP Per Capita",
        "SCOE Elasticity of Per Household Electrical Applicance Demand to GDP Per Capita",
        "SCOE Elasticity of Per Household Heat Energy Demand to GDP Per Capita"
    ]

    out = _build_uncertainty_from_variable(
        modvars,
        (0.75, 1.25),
        trajgroup,
        model_attributes,
        field_max,
        field_min,
        field_trajgroup,
        field_variable,
        **kwargs,
    )
    
    return out




def _build_uncertainty_storage_costs(
    trajgroup: int,
    model_attributes: 'ModelAttributes',
    field_max: str,
    field_min: str,
    field_trajgroup: str,
    field_variable: str,
    transformers: Union['Transformers', None] = None,
    **kwargs,
) -> pd.DataFrame:
    """Build uncertainties for energy storage

    Function Arguments
    ------------------
    trajgroup : int
        trajectory group to assign
    field_max : str
        Field used to store max range scalar in templates
    field_min : str
        Field used to store min range scalar in templates
    field_trajgroup : str
        Field storing the trajectory group
    field_variable : str
        Field used to store the variable field name
    """
    
    # variables to tie into groups
    out = _build_uncertainty_from_variable(
        "NemoMod CapitalCostStorage",
        (0.8, 1.2),
        trajgroup,
        model_attributes,
        field_max,
        field_min,
        field_trajgroup,
        field_variable,
        **kwargs,
    )
    
    return out




def _build_uncertainty_technology_costs(
    trajgroup: int,
    model_attributes: 'ModelAttributes',
    field_max: str,
    field_min: str,
    field_trajgroup: str,
    field_variable: str,
    transformers: Union['Transformers', None] = None,
    **kwargs,
) -> pd.DataFrame:
    """Build uncertainties for fuels. 

    Function Arguments
    ------------------
    trajgroup : int
        trajectory group to assign
    field_max : str
        Field used to store max range scalar in templates
    field_min : str
        Field used to store min range scalar in templates
    field_trajgroup : str
        Field storing the trajectory group
    field_variable : str
        Field used to store the variable field name
    """
    if transformers is None:
        return (None, trajgroup)

    dict_tech_classes = transformers.model_enerprod.get_tech_info_dict()
    _CATS_ETNC_PP = dict_tech_classes.get("all_techs_pp")

    groups = [
        [
            "pp_coal",
            "pp_coal_ccs",
            "pp_gas",
            "pp_gas_ccs",
            "pp_oil"
        ],
        [
            "pp_nuclear"
        ],
        [
            "pp_geothermal",
            "pp_hydropower",
            "pp_solar",
            "pp_wind",
            
        ]
    ]

    scalar_ranges = [(0.75, 1.25), (0.75, 1.5), (1.0, 2.0)]


    # variables to tie into groups
    modvar_capex = model_attributes.get_variable("NemoMod CapitalCost")
    modvar_fixed = model_attributes.get_variable("NemoMod FixedCost")
    modvar_variable = model_attributes.get_variable("NemoMod VariableCost")

    df_out = []

    for i, grp in enumerate(groups):

        # group the model variables together
        for modvar in [
            modvar_capex,
            modvar_fixed,
            modvar_variable
        ]:
        
            ranges = scalar_ranges[i]
            fields_combine = modvar.build_fields(category_restrictions = grp, )
            
            
            df_component = pd.DataFrame(
                {
                    field_variable: fields_combine,
                    field_max: [ranges[1] for x in fields_combine],
                    field_min: [ranges[0] for x in fields_combine],
                    field_trajgroup: [trajgroup for x in fields_combine],
                }
            )

            df_out.append(df_component)

        trajgroup += 1

    
    df_out = sf._concat_df(df_out)
    out = (df_out, trajgroup, )
    
    return out

    out = (df_component, trajgroup, )
    
    return out



def _build_uncertainty_transportation_elasticities(
    trajgroup: int,
    model_attributes: 'ModelAttributes',
    field_max: str,
    field_min: str,
    field_trajgroup: str,
    field_variable: str,
    transformers: Union['Transformers', None] = None,
    **kwargs,
) -> pd.DataFrame:
    """Build uncertainties for energy storage

    Function Arguments
    ------------------
    trajgroup : int
        trajectory group to assign
    field_max : str
        Field used to store max range scalar in templates
    field_min : str
        Field used to store min range scalar in templates
    field_trajgroup : str
        Field storing the trajectory group
    field_variable : str
        Field used to store the variable field name
    """
    
    modvars = [
        "Elasticity of Megatonne-Kilometer Demand to GDP",
        "Elasticity of Passenger-Kilometer Demand per Capita to GDP per Capita"
    ]

    out = _build_uncertainty_from_variable(
        modvars,
        (0.75, 1.25),
        trajgroup,
        model_attributes,
        field_max,
        field_min,
        field_trajgroup,
        field_variable,
        each_modvar_gets_unique_trajgroup = True,
        **kwargs,
    )
    
    return out




def get_max_min_field(
    strategies: 'Strategies',
    tp: int,
    str_replace: str = "(\\d*$)",
) -> Tuple[str, str]:
    """Get the fields storing the max/min ranges
    """
    pattern_max = (
        strategies
        .input_template
        .regex_template_max
        .pattern
        .replace(str_replace, str(tp), )
    )
    
    pattern_min = (
        strategies
        .input_template
        .regex_template_min
        .pattern
        .replace(str_replace, str(tp), )
    )
    
    out = (pattern_max, pattern_min, )

    return out




def main(
   strategies: 'Strategies', 
) -> pd.DataFrame:
    """Build complete uncertainties and trajectory groups data
    """

    ##  INITIALIZATION 

    model_attributes = strategies.model_attributes
    time_periods = sc.TimePeriods(model_attributes) 

    # get fields--start with max time period
    tp_max = max(time_periods.all_time_periods)
    field_max, field_min = get_max_min_field(strategies, tp_max, )
    field_trajgrup = strategies.input_template.field_req_variable_trajectory_group
    field_variable = strategies.input_template.field_req_variable


    # arguments to pass to elemental dfs
    args = (
        model_attributes,
        field_max,
        field_min,
        field_trajgrup,
        field_variable
    )

    df_out = []


    ##  ITERATE OVER FUNCTIONS

    funcs = [
        _build_uncertainty_fuel_exports,
        _build_uncertainty_fuel_prices,
        _build_uncertainty_gdp,
        _build_uncertainty_inen_demand_scalar,
        _build_uncertainty_inen_elasticities,
        _build_uncertainty_population,
        _build_uncertainty_scoe_elasticities,
        _build_uncertainty_storage_costs,
        _build_uncertainty_technology_costs,
        _build_uncertainty_transportation_elasticities
    ]


    # initialize; this is updated within each function, 
    # which returns the next trajgroup to pass to the next function
    trajgroup = 1

    for _, func in enumerate(funcs):
        df_cur, trajgroup = func(
            trajgroup, 
            *args, 
            transformers = strategies.transformations.transformers, 
        )
        df_out.append(df_cur, )

    df_out = sf._concat_df(df_out, )

    return df_out
