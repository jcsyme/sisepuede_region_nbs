###  Build a table mapping transformations to sample groups

"""
##===================
# get space of transformation codes to restrict to
strat = strategies.get_strategy(6002)            # all actions
strat_transformations = [x.code for x in strat.get_transformation_list(
    transformations.all_transformation_codes,
    transformations
)]


df_base = transformations.get_transformation(transformations.code_baseline, )
df_base = df_base()

dict_transformation_codes_by_field = {}

for tf_code in strat_transformations:#transformations.all_transformation_codes:
    tr_cur = transformations.get_transformation(tf_code, )
    df_cur = tr_cur()

    # fields the 
    fields = []

    for field in df_cur.columns:
        test = all(df_cur[field] == df_base[field])
        if test: continue
        
        # check if codes are specified
        codes = dict_transformation_codes_by_field.get(field, )
        if codes is None:
            dict_transformation_codes_by_field.update({field: [tf_code] })
        else:
            dict_transformation_codes_by_field[field].append(tf_code)
            
dict_transformation_codes_by_field = dict((k, "|".join(v)) for k, v in dict_transformation_codes_by_field.items())


##  MERGE CODES AND SAMPLE GROUPS TOGETHER

df_code_by_field = pd.DataFrame(
    dict_transformation_codes_by_field.items(), 
    columns = ["variable_field", "transformation_code"],
)

df_varspec_to_sg = pd.read_excel(
    "/Users/usuario/Documents/Projects/Louisiana/merges/20250521/variable_specification_to_sample_group.xlsx"
)

df_map = pd.merge(
    df_varspec_to_sg,
    df_code_by_field.rename(
        columns = {"variable_field": "variable_specification", }
    ),
    how = "left"
)

df_map = pd.merge(
    df_map,
    transformations.attribute_transformation.table
    .get(
        ["transformation_code", "transformation_name"]
    ),
    how = "left"
)


df_map.to_csv(
    "/Users/usuario/Documents/Projects/Louisiana/merges/20250521/variable_specification_to_sample_group_and_transformation_code.csv",
    encoding = "UTF-8",
    index = None,
)

===================##
"""