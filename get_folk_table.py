from folktables import (
    ACSDataSource,
    ACSIncome,
    ACSMobility,
    ACSPublicCoverage,
    ACSEmployment,
    ACSTravelTime,
    generate_categories,
)

survey_year="2017"

data_source = ACSDataSource(survey_year=survey_year, horizon="1-Year", survey="person")
ca_data = data_source.get_data(states=["CA"], download=True)

definition_df = data_source.get_definitions(download=True)

for task, task_fun in zip(
    # ["income", "public_coverage", "mobility", "employ"],
    # [ACSIncome, ACSPublicCoverage, ACSMobility, ACSEmployment],
    ["income", "employ"],
    [ACSIncome, ACSEmployment],
):
    categories = generate_categories(
        features=ACSIncome.features, definition_df=definition_df
    )

    features, targets, _ = task_fun.df_to_pandas(
        ca_data, categories=categories, dummies=True
    )
    
    features.to_csv(f"ca_{task}_{survey_year}_features.csv", index=False)
    targets.to_csv(f"ca_{task}_{survey_year}_labels.csv", index=False)
    
