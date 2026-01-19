import _iea_util as ieau
import numpy as np
import pandas as pd
import pathlib
import re
import sisepuede.core.model_attributes as ma
import sisepuede.pipeline.projection_utilities as putil
import sisepuede.utilities._toolbox as sf
from typing import *

##  SOME GLOBALS

# fields
_FIELD_CW_IEA_FUELS = ieau._FIELD_CW_IEA_FUELS
_FIELD_CW_IEA_FUEL_GROUPS = ieau._FIELD_CW_IEA_FUEL_GROUPS
_FIELD_IEA_FUEL = "electricity generation sources in Mexico"
_FIELD_IEA_UNITS = ieau._FIELD_IEA_UNITS
_FIELD_IEA_VALUE = ieau._FIELD_IEA_VALUE
_FIELD_IEA_YEAR = ieau._FIELD_IEA_YEAR

# regular expressions
_REGEX_IEA_FN_CONSUMPTON = re.compile("International Energy Agency - (.\D*) total final consumption by source in (.\D*).csv")



##  FUNCTIONS



def get_iea(
    path_iea: pathlib.Path,
    dict_iea_fuels: Dict[str, str],
    time_periods: 'TimePeriods',
    models: 'SISEPUEDEModels',
    convert_fuel_to_tech: bool = True,
    dict_category_replacements: Union[Dict[str, str], None] = None,
    field_cw_iea_fuels: str = _FIELD_CW_IEA_FUELS,
    field_cw_iea_fuel_groups: str = _FIELD_CW_IEA_FUEL_GROUPS,
    field_iea_fuel: str = _FIELD_IEA_FUEL,
    field_iea_units: str = _FIELD_IEA_UNITS,
    field_iea_value: str = _FIELD_IEA_VALUE,
    field_iea_year: str = _FIELD_IEA_YEAR,
    field_total_base: str = "total",
    fuels_iea_keep: Union[List[str], None] = None,
    modvar: Union['ModelVariable', None] = None,
    normalize: bool = False,
) -> dict:
    """get the dictionary mapping IEA fuels to SSP fuels

    Function Arguments
    ------------------

    Keyword Arguments
    -----------------
    dict_category_replacements : Union[Dict[str, str], None]
        Category replacements to map  
    fuels_iea_keep: Union[List[str], None] = None,
    modvar: Union['ModelVariable', None] = None,
    """

    df_iea = pd.read_csv(path_iea, )
    
    out = ieau.get_iea_df_for_ssp(
        df_iea,
        dict_iea_fuels,
        time_periods,
        models,
        field_iea_fuel,
        convert_fuel_to_tech = convert_fuel_to_tech,
        dict_category_replacements = dict_category_replacements,
        field_cw_iea_fuels = field_cw_iea_fuels,
        field_cw_iea_fuel_groups = field_cw_iea_fuel_groups,
        field_iea_units = field_iea_units,
        field_iea_value = field_iea_value,
        field_iea_year = field_iea_year,
        field_total_base = field_total_base,
        fuels_iea_keep = fuels_iea_keep,
        modvar = modvar,
        normalize = normalize,
    )
    
    return out
    


def overwrite_msp_from_iea(
    df_input: pd.DataFrame,
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
    df_cw = pd.read_csv(path_iea_cats, )
    df_cw, dict_iea_fuels = ieau.get_iea_dict(
        df_cw, 
        models.model_attributes, 
    )

    #
    (
        df_elec,
        df_total,
        field_total,
        units_iea,
    ) = get_iea(
        path_iea,
        dict_iea_fuels,
        time_periods,
        models,
        modvar = modvar_msp,
        normalize = True,
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