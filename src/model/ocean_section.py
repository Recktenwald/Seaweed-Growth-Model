"""
File contains the class OceanSection, which is used to represent
a section of the ocean. This can be either a large marine ecosystem
or simply a part of a global grid.
"""
import pandas as pd
import numpy as np
from dataenforce import Dataset, validate

from src.model import seaweed_growth as sg

RawOceanSections = Dataset["name", "salinity", "temperature",  # I would suggest using an ID instad of name
                           "nitrate", "ammonium", "phosphate", "illumination"]

OceanSections = Dataset[RawOceanSections, "salinity_factor", "nutrient_factor",
                        "illumination_factor", "temp_factor", "seaweed_growth_rate", "months_since_war"]


# This is probably not a good way, but I don't know the dataenforce package that well
# There is probably a better way to check if something fits the description
# But this decorator simply makes the function call _fail_ if it does not. So good enough for now.
@validate
def is_raw_ocean_section(df: RawOceanSections):
    return(True)


@validate
def is_ocean_section(df: OceanSections):
    return(True)


def calculate_factors(df : RawOceanSections) -> OceanSections:
    """
    Calculates the factors and growth rate for the ocean section
    """

    # First we compute the various factors
    df["salinity_factor"] = df["salinity"].map(sg.salinity_single_value)
    df["nutrient_factor"] = np.vectorize(sg.nutrient_single_value)(
        df["nitrate"], df["ammonium"], df["phosphate"])
    # Alternatively without numpy
    # df["nutrient_factor"] = df.apply(
    #     lambda x: sg.nutrient_single_value(x["nitrate"], x["ammonium"], x["phosphate"]),
    #     axis=1,
    # )
    df["illumination_factor"] = df["illumination"].map(sg.illumination_single_value)
    df["temp_factor"] = df["temperature"].map(sg.temperature_single_value)

    # Now we compute the growth rate
    df["seaweed_growth_rate"] = df.apply(
        lambda x: sg.growth_factor_combination_single_value(
            x["illumination_factor"],
            x["temp_factor"],
            x["nutrient_factor"],
            x["salinity_factor"],
        ),
        axis=1,
    )

    # Months since war
    df["months_since_war"] = df \
        .groupby("name") \
        .apply(lambda x: range(-3, x.shape[0] - 3, 1)) \
        .explode() \
        .values \
        .astype("int64")  # for some reason this is needed to surpress a pandas "future warning" later.

    return(df)

    # Given that the rewrite we don't have a wrapper class anymore, but simply a data frame, I think it makes sense
    # to not have specific functions for computing the mean or selecting a month.
    # This can now easily be done directly with pandas.


class OceanSection:
    """
    Class the represents a section of the ocean.
    calculates for every section how quickly seaweed can grow
    and also saves the single factors for growth
    """

    def __init__(self, name, data):
        # Add the name
        self.name = name
        # Add the data
        self.salinity = data["salinity"]
        self.temperature = data["temperature"]
        self.nitrate = data["nitrate"]
        self.ammonium = data["ammonium"]
        self.phosphate = data["phosphate"]
        self.illumination = data["illumination"]
        # Add the factors
        self.salinity_factor = None
        self.nutrient_factor = None
        self.illumination_factor = None
        self.temp_factor = None
        self.seaweed_growth_rate = None
        # Add the dataframe
        self.section_df = None

    def calculate_factors(self):
        """
        Calculates the factors and growth rate for the ocean section
        Arguments:
            None
        Returns:
            None
        """
        # Calculate the factors
        self.salinity_factor = sg.calculate_salinity_factor(self.salinity)
        self.nutrient_factor = sg.calculate_nutrient_factor(
            self.nitrate, self.ammonium, self.phosphate
        )
        self.illumination_factor = sg.calculate_illumination_factor(self.illumination)
        self.temp_factor = sg.calculate_temperature_factor(self.temperature)

    def calculate_growth_rate(self):
        """
        Calculates the growth rate for the ocean section
        Arguments:
            None
        Returns:
            None
        """
        # Calculate the growth rate
        self.seaweed_growth_rate = sg.growth_factor_combination(
            self.illumination_factor,
            self.temp_factor,
            self.nutrient_factor,
            self.salinity_factor,
        )

    def create_section_df(self):
        """
        Creates a dataframe that contains all the data for a given section
        This can only be run once the factors have been calculated
        """
        # check if the factors have been calculated
        assert self.salinity_factor is not None
        assert self.nutrient_factor is not None
        assert self.illumination_factor is not None
        assert self.temp_factor is not None
        assert self.seaweed_growth_rate is not None

        # Create the dataframe
        section_df = pd.DataFrame(
            {
                "salinity": self.salinity,
                "temperature": self.temperature,
                "nitrate": self.nitrate,
                "ammonium": self.ammonium,
                "phosphate": self.phosphate,
                "illumination": self.illumination,
                "salinity_factor": self.salinity_factor,
                "nutrient_factor": self.nutrient_factor,
                "illumination_factor": self.illumination_factor,
                "temp_factor": self.temp_factor,
                "seaweed_growth_rate": self.seaweed_growth_rate,
            }
        )
        # Add a column with the month since war
        section_df["months_since_war"] = list(range(-3, section_df.shape[0] - 3, 1))
        section_df.set_index("months_since_war", inplace=True)
        # Add the dataframe to the class
        section_df.name = self.name
        self.section_df = section_df

    def calculate_mean_growth_rate(self):
        """
        Calculates the mean growth rate and returns it
        """
        # check if the dataframe has been created
        assert self.section_df is not None
        # calculate the mean growth rate
        return self.section_df["seaweed_growth_rate"].mean()

    def select_section_df_date(self, month):
        """
        Selectes a date from the section df and returns it
        Arguments:
            date: the date to select
        Returns:
            the dataframe for the date
        """
        # check if the dataframe has been created
        assert self.section_df is not None
        # select the dataframe for the date
        return self.section_df.loc[month, :]
