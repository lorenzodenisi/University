import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import colorsys
import seaborn as sns
import numpy as np


def main():
    ny_poi = pd.read_csv("pois_all_info", sep='\t')
    district_poi = pd.read_csv("ny_municipality_pois_id.csv", sep='\t', names=['@id'])

    ny_dist_pois = pd.DataFrame.join(ny_poi, district_poi, rsuffix="_right")
    ny_dist_pois = ny_dist_pois.drop(columns=["@id_right"])

    print(ny_dist_pois.dtypes)
    print(ny_dist_pois.isnull().sum())

    poi_categories = ["amenity", "shop", "public_transport", "highway"]

    plot_category_hist(ny_dist_pois, categories=poi_categories)

    display_category(ny_dist_pois, poi_categories, "New_York_City_map.PNG")

    n_bins = 20  # number of ranges for both latitude and longitude
    ny_dist_pois["grid_id"] = binning2D(ny_dist_pois, "@lon", n_bins, "@lat", n_bins)
    count_per_type = get_poi_bins(ny_dist_pois, n_bins, n_bins, poi_categories)
    pairwise_correlation(count_per_type, poi_categories)


def plot_category_hist(df, categories):
    for category in categories:
        pois = df[~df[category].isnull()]
        pois[category].value_counts().nlargest(10).plot.barh()  # limit to 10 largest types
        plt.title(str(category))
        plt.show()


def display_category(df, categories, map_path):
    img = plt.imread(map_path)
    plt.figure(figsize=(10, 8))
    extent = [-74.262612, -73.670997, 40.493287, 40.916527]

    colors = sns.color_palette(None, len(categories))

    plt.imshow(img, zorder=0, extent=extent)
    for i, category in enumerate(categories):
        pois = df[~df[category].isnull()]
        plt.scatter(pois['@lon'], pois["@lat"], c=[colors[i], ], s=0.5, alpha=0.5, label=category)
    plt.legend()
    plt.show()


def binning2D(df, dim1, n_bins1, dim2, n_bins2):
    filtered_df = df[~df[dim1].isnull()]
    filtered_df = filtered_df[~filtered_df[dim2].isnull()]

    # binning
    # labels=False is used to get bin index instead of bin range
    bin1_codes = pd.cut(filtered_df[dim1], n_bins1, labels=False).astype(int)
    bin2_codes = pd.cut(-filtered_df[dim2], n_bins2, labels=False).astype(int) # minus used to invert binning index order

    grid_index = bin1_codes + bin2_codes * n_bins1  # encode the two bin indexes into matrix index

    return grid_index


def get_poi_bins(df, n_bin_x, n_bin_y, categories):
    n_bins = n_bin_x * n_bin_y

    poi_per_types = {}
    for category in categories:
        filtered_df = df[~df[category].isnull()]
        filtered_df = filtered_df[~filtered_df["grid_id"].isnull()]

        # heatmap per category
        values = filtered_df["grid_id"].astype(int).value_counts().to_dict()
        for i in range(n_bins):
            if i not in values.keys(): values[i] = 0

        sorted_values = [values[i] for i in sorted(values.keys())]

        grid = np.reshape(sorted_values, (n_bin_x, n_bin_y))
        plt.imshow(grid, cmap='hot', interpolation='nearest')
        plt.title(category)
        plt.show()

        # counts per type
        types = filtered_df[category].value_counts().nlargest(10).axes[0]  # limit to 10 largest types
        count = pd.DataFrame()
        for type in types:
            count[type] = filtered_df[filtered_df[category] == type]["grid_id"].value_counts().fillna(0)
        pass
        poi_per_types[category] = count

    return poi_per_types


def pairwise_correlation(count_per_type, categories):
    df = pd.DataFrame()
    plt.figure(figsize=(12, 10))

    # concat used to combine two dataframe of two different categories according to bin (coords range)
    for category in categories:
        df = pd.concat([df, count_per_type[category]], axis=1)
    sns.heatmap(df.corr())
    plt.show()


if __name__ == "__main__":
    main()
