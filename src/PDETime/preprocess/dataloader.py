from typing import Optional, Union, List
from torch.utils.data import Dataset

from torch import Tensor
from pandas import DataFrame, Series, to_datetime, to_numeric
from sklearn.preprocessing import MinMaxScaler

class PDETimeDataset(Dataset):

    TEMPORAL_FEATURE_NAMES = ('year', 'quarter', 'month', 'day_of_year', 'day_of_week', 'hour', 'minute', 'second')

    def __init__(self, df:DataFrame, horizon:int=96, lookback:Optional[int]=None, time_index_name:str='date', temporal_feature_names:Optional[List[str]]=None):
        self.df = df
        self.horizon = horizon
        self.lookback = lookback if lookback else 5 * horizon
        self.time_index_name = time_index_name

        self.scaler = MinMaxScaler()
        self.temporal_feature_names = temporal_feature_names

        self.make_temporal_features(self.time_index_name)
        self.scale_data()

        self.features = self.df.copy().drop(columns=self.time_index_name).drop(columns=self.temporal_feature_names).values
        self.time_index = self.df.copy()[self.time_index_name].values
        self.temporal_features = self.df.copy()[self.temporal_feature_names].values

    def __len__(self):
        return len(self.df) - self.lookback - self.horizon + 1

    def __getitem__(self, idx:int): 
        #TODO: accept slices too
        print(idx, len(self))
        if idx >= len(self):
            raise IndexError("Index out of bounds (valid indices are [0:{}]".format(len(self)-1) + " but got index {}".format(idx) + " instead)")
        x = self.features[idx : idx + self.lookback]
        t = self.temporal_features[idx : idx + self.lookback + self.horizon]
        tau = self.time_index[idx : idx + self.lookback + self.horizon]
        labels = self.features[idx : idx + self.lookback + self.horizon]
        return x, t, tau, labels
    
    def make_temporal_features(self, time_index_name:Optional[str]=None) -> None:
        """
        Adds temporal features to the given DataFrame based on the specified time index column.
        If the class is initialised with temporal_feature_names, only those features are added.
        Otherwise, useful temporal features are added automatically.

        Args:
            time_index (str): The name of the column containing the time index. Default is 'date'.
        """
        time_index_name = time_index_name if time_index_name else self.time_index_name
        self.df[time_index_name] = to_datetime(self.df[time_index_name])

        if self.temporal_feature_names is None:
            self.temporal_feature_names = self.choose_temporal_features(self.df, time_index_name)
        for feature in self.temporal_feature_names:
            self.df[feature] = self.df[time_index_name].dt.__getattribute__(feature)
        
    @staticmethod
    def choose_temporal_features(df, time_index_name:str='date') -> List[str]:
        """
        Automatically selects the most useful temporal features from the DataFrame.
        Call this before doing train-test split so that features are consistent across splits.
        Args:
            df (DataFrame): The input DataFrame.
            time_index_name (str): The name of the column containing the time index. Default is 'date'.

        Returns:
            List[str]: The list of selected temporal features.
        """
        df[time_index_name] = to_datetime(df[time_index_name])
        temporal_feature_names = []
        for feature in PDETimeDataset.TEMPORAL_FEATURE_NAMES:
            values = df[time_index_name].dt.__getattribute__(feature)
            # Only add varying calendar features to the DataFrame. Previously
            # constant features (e.g. 'second' for ETTm1, also 'minute' for
            # ETTh) were added but excluded from temporal_feature_names.
            # They then leaked into self.features as extra target channels.
            if values.nunique() > 1:
                df[feature] = values
                temporal_feature_names.append(feature)
        return temporal_feature_names
        

    def scale_data(self) -> None:
        """
        Scales the input DataFrame using MinMaxScaler, in place.
        """
        self.df = self.df.apply(to_numeric)
        self.df[self.temporal_feature_names + [self.time_index_name]] = self.scaler.fit_transform(self.df[self.temporal_feature_names + [self.time_index_name]])

    def unscale_data(self, df: Optional[Union[DataFrame, Tensor]]) -> DataFrame:
        """
        Unscales the input DataFrame using MinMaxScaler.

        Args:
            df (DataFrame): The input DataFrame to be unscaled. In the order of temporal features and time index.

        Returns:
            DataFrame: The unscaled DataFrame.
        """
        df = df if df else self.df[self.temporal_feature_names + [self.time_index_name]]
        return self.scaler.inverse_transform(df)
    
    @classmethod
    def train_test_val_split(cls, df:DataFrame, horizon:int=96, lookback:Optional[int]=None, time_index_name:str='date', temporal_feature_names:Optional[List[str]]=None, val_ratio:float=0.2, test_ratio:float=0.2):
        """
        Creates three separate datasets for training, testing, and validation from the input DataFrame.
        Datasets will all have the same features so model can be trained on one and tested on another.

        Args:
            df (DataFrame): The input DataFrame containing the data.
            horizon (int): The number of time steps to predict into the future.
            lookback (Optional[int]): The number of previous time steps to consider for prediction. If None, all previous time steps will be used.
            time_index_name (str): The name of the column in the DataFrame that represents the time index.
            temporal_feature_names (Optional[List[str]]): The names of the columns in the DataFrame that represent the temporal features. If None, all columns except the time index column will be considered as temporal features.
            val_ratio (float): The ratio of data to be used for validation.
            test_ratio (float): The ratio of data to be used for testing.

        Returns:
            Tuple[PDETimeDataset, PDETimeDataset, PDETimeDataset]: A tuple containing the train, test, and validation datasets.

        """
        temporal_feature_names = cls.choose_temporal_features(df, time_index_name) if not temporal_feature_names else temporal_feature_names

        total_length = len(df)
        train_length = int(total_length * (1 - val_ratio - test_ratio))
        val_length = int(total_length * val_ratio)
        train_data = df[:train_length]
        val_data = df[train_length:train_length+val_length]
        test_data = df[train_length+val_length:]

        train_dataset = cls(
            df=train_data,
            horizon=horizon,
            lookback=lookback,
            time_index_name=time_index_name,
            temporal_feature_names=temporal_feature_names
            )

        test_dataset = cls(
            df=test_data,
            horizon=horizon,
            lookback=lookback,
            time_index_name=time_index_name,
            temporal_feature_names=temporal_feature_names
            )

        val_dataset = cls(
            df=val_data,
            horizon=horizon,
            lookback=lookback,
            time_index_name=time_index_name,
            temporal_feature_names=temporal_feature_names
            )

        return train_dataset, test_dataset, val_dataset