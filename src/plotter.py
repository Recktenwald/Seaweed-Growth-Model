from src.seaweed_model import SeaweedModel

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


class Plotter:
    """
    Class to organize all the plotting functions
    """

    def __init__(self, seaweed_model):
        self.seaweed_model = seaweed_model

    def plot_growth_rate_by_lme_bar(self, date, path=""):
        """
        Plots the growth rate for the model based on LME as a bar chart
        """
        assert self.seaweed_model.lme_or_grid == "lme"
        date_section_df = self.seaweed_model.construct_df_from_sections_for_date(date)
        ax = date_section_df.seaweed_growth_rate.sort_values().plot(kind="bar")
        ax.set_title("Growth Rate by LME")
        ax.set_xlabel("LME")
        ax.set_ylabel("Fraction of optimal growth rate")
        fig = plt.gcf()
        fig.set_size_inches(10, 5)
        plt.savefig(
            path + "growth_rate_by_lme_bar" + str(date) + ".png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()

    def plot_growth_rate_by_lme_global(self, date, path=""):
        """
        Plots the growth rate fraction for all LME on a global map
        """
        assert self.seaweed_model.lme_or_grid == "lme"
        date_section_df = self.seaweed_model.construct_df_from_sections_for_date(date)
        lme_shape = gpd.read_file("data/lme_shp/lme66.shp")
        lme_global = lme_shape.merge(
            date_section_df, left_on="LME_NUMBER", right_index=True
        )
        ax = lme_global.plot(
            column="seaweed_growth_rate",
            cmap="Greens",
            legend=True,
            edgecolor="black",
            linewidth=0.1,
            vmin=0,
            vmax=0.8,
        )
        ax.set_title("Fraction of optimal growth rate on date: " + str(date))
        fig = plt.gcf()
        fig.set_size_inches(10, 5)
        plt.savefig(
            path + "growth_rate_by_lme_global_" + str(date) + ".png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()

    def calculate_mean_groth_rate_by_lme(self):
        """
        Calculates the mean growth rate for a LME for the whole
        time period modelled.
        Arguments:
            None
        Returns:
            None
        """
        assert self.seaweed_model.lme_or_grid == "lme"
        growth_rate_dict = {}
        for section_name, section_instance in self.seaweed_model.sections.items():
            growth_rate_dict[
                section_name
            ] = section_instance.calculate_mean_growth_rate()
        growth_df = pd.DataFrame.from_dict(growth_rate_dict, orient="index")
        growth_df.columns = ["mean_growth_rate"]
        print("LMEs with highest mean growth rates fraction are:")
        print(growth_df["mean_growth_rate"].sort_values(ascending=False).head())
        print(
            "Global mean growth rate fraction over the whole time period:"
            + str(growth_df["mean_growth_rate"].mean())
        )

    def plot_growth_rate_by_best_lme_as_line(self, path="", window=10):
        """
        Takes the growthrate of the 3 best LMEs (by mean growth rate)
        29    0.391329
        11    0.309920
        38    0.304329
        and plots them over time.

        Arguments:
            path: the path to save the plot to
            window: the window size for the rolling mean

        Returns:
            None
        """
        assert self.seaweed_model.lme_or_grid == "lme"
        best_lme = [29, 11, 38]
        best_lme_dict = {
            29: "Benguela Current",
            11: "Pacific Central-American Coastal",
            38: "Indonesian Sea",
        }
        ax = plt.gca()
        df_list = []
        for section_name, section_instance in self.seaweed_model.sections.items():
            if section_name in best_lme:
                temp_df = (
                    section_instance.section_df["seaweed_growth_rate"]
                    .rolling(window=window)
                    .mean()
                    .dropna()
                )
                temp_df.name = best_lme_dict[section_name]
                df_list.append(temp_df)
        mean_df = pd.concat(df_list, axis=1)
        mean_df.plot(ax=ax, alpha=0.8)
        ax.set_title("Smoothed Growth Rate of the 3 most productive LMEs")
        ax.set_ylabel("Fraction of optimal growth rate")
        fig = plt.gcf()
        fig.set_size_inches(10, 5)
        plt.savefig(path + "growth_rate_by_best.png", dpi=200, bbox_inches="tight")
        plt.close()


def main():
    model = SeaweedModel()
    model.add_data_by_lme(
        [i for i in range(1, 67)], "data/seaweed_environment_data_in_nuclear_war.csv"
    )
    model.calculate_factors()
    model.calculate_growth_rate()
    model.create_section_dfs()
    # Plot for a selection of dates
    # beginning of simulation before nuclear war
    # one month after nuclear war
    # one year after nuclear war
    # two years after nuclear war
    # and so forth
    dates = (
        ["2001-01-31"]
        + ["200" + str(i) + "-06-30" for i in range(2, 10)]
        + ["20" + str(i) + "-06-30" for i in range(10, 18)]
    )
    plotter = Plotter(model)
    for date in dates:
        plotter.plot_growth_rate_by_lme_bar(date, path="results/lme/")
        plotter.plot_growth_rate_by_lme_global(date, path="results/lme/")

    # Print the best 3 LMEs by mean growth rate
    plotter.calculate_mean_groth_rate_by_lme()

    # Plot the growth rate of the 5 best LMEs
    plotter.plot_growth_rate_by_best_lme_as_line(path="results/lme/", window=1)


if __name__ == "__main__":
    main()