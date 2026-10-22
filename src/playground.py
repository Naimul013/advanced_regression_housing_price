from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

def ordinal_pipeline(features: list[str], order: list[str]):
    pipeline = Pipeline(
        steps=[
            (
                'imputer',
                SimpleImputer(
                    strategy='constant',
                    fill_value='None'
                )
            ),
            (
                'encoder',
                OrdinalEncoder(
                    categories=[order] * len(features),
                    handle_unknown='use_encoded_value',
                    unknown_value=-1
                )
            )
        ]
    )

    return pipeline
    