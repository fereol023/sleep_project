import pandas as pd
from itertools import product
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from dstoolbox.pipeline import DataFrameFeatureUnion
from sklearn.pipeline import Pipeline


class FloatEncoder(BaseEstimator, TransformerMixin):
    """
    Ce transformer va caster les variables "to_cast_in_float" en float 
    """
    def __init__(self, columns_to_transform = None):
        super().__init__()
        self.columns_to_transform = columns_to_transform

    def fit(self, X, y = None):
        if self.columns_to_transform is not None :
            self.X = X[self.columns_to_transform]
        return self # on n'estime rien - limite on peut utiliser assert X[col] is not str pour preparer le codage au float

    def transform(self, X):
        for col in self.columns_to_transform:
            self.X[col] = self.X[col].astype(float)
        return self.X
    
class IntEncoder(BaseEstimator, TransformerMixin):
    """
    Ce transformer va caster les variables "to_cast_in_float" en float 
    """
    def __init__(self, columns_to_transform = None):
        super().__init__()
        self.columns_to_transform = columns_to_transform

    def fit(self, X, y = None):
        if self.columns_to_transform is not None :
            self.X = X[self.columns_to_transform]
        return self 
    
    def transform(self, X):
        for col in self.columns_to_transform:
            self.X[col] = self.X[col].astype(int)
        return self.X

class OneHotEncoderPandas(BaseEstimator, TransformerMixin):
    """
    Cette implémentation du OneHotEncoder renvoie un DataFrame en sortie et non un array numpy. Ceci est fait pour garantir la
    traçabilité des variables et faciliter l'interprétation du modèle dans la suite.
    """
    def __init__(self, columns = None):
        super().__init__()
        self.columns = columns
        self.ohe = OneHotEncoder(handle_unknown= "ignore", sparse = False) #, drop='first')

    def fit(self, X, y = None):
        if self.columns is not None:
            self.X = X[self.columns]
        # on fit l'encoder one hot
        self.ohe.fit(self.X)

        # Récupération des noms de colonnes + categories pour l'interprétabilité
        # Ex : "situation_-1", "situation_10", etc..
        self.feature_category_pairs = []
        for i, feature in enumerate(self.ohe.feature_names_in_):
            feature_category = product([feature], self.ohe.categories_[i])
            for pair in feature_category:
                self.feature_category_pairs.append("{}_{}".format(*pair))
        return self
    
    def transform(self, X):
        if self.columns is not None:
            self.X = X[self.columns]
            
        return pd.DataFrame(self.ohe.transform(self.X), columns = self.feature_category_pairs)


class OrdinalEncoderPandas(BaseEstimator, TransformerMixin):
    """
    Cette implémentation du OrdinalEncoder renvoie un DataFrame en sortie et non un array numpy. Ceci est fait pour garantir la
    traçabilité des variables et faciliter l'interprétation du modèle dans la suite.
    """
    def __init__(self, columns = None):
        super().__init__()
        self.columns = columns
        self.orde = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=999, encoded_missing_value=9999)

    def fit(self, X, y = None):
        if self.columns is not None :
            self.X = X[self.columns]
        
        self.orde.fit(self.X) # ICI
        return self

    def transform(self, X):
        if self.columns is not None:
            X = X[self.columns]      # ICI
        return pd.DataFrame(self.orde.transform(X), columns = self.columns)
        
