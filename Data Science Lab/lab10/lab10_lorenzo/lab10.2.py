import pandas as pd
import matplotlib.pyplot as plt
import datetime
import seaborn as sns


def main():
    data = pd.read_csv("831394006_T_ONTIME.csv", parse_dates=[0])
    print(data.describe())
    data.info()

    uniques_carriers = data.UNIQUE_CARRIER.unique()
    print(f"Unique carriers: {len(uniques_carriers)}, {uniques_carriers}")

    data = data[data.CANCELLED == 0]

    print("\nNumber of flights per carrier:")
    print(data.groupby("UNIQUE_CARRIER").FL_NUM.count())

    print("\nAverage delay per carrier:")
    print(data.groupby("UNIQUE_CARRIER").DEP_DELAY.mean() - data.groupby("UNIQUE_CARRIER").ARR_DELAY.mean())

    data["weekday"] = data.FL_DATE.apply(lambda x: x.isoweekday())
    data["delaydelta"] = data["ARR_DELAY"] - data["DEP_DELAY"]
    data.groupby("weekday").ARR_DELAY.mean().plot()
    plt.xlabel("day of the week")
    plt.ylabel("average arrival delay")
    plt.title("AVG DELAY vs DOW")
    plt.grid()
    plt.show()

    wknd = data[data.weekday > 5].groupby("UNIQUE_CARRIER").ARR_DELAY.mean().rename("Weekend")
    fer = data[data.weekday <= 5].groupby("UNIQUE_CARRIER").ARR_DELAY.mean().rename("Workday")

    pd.concat([wknd, fer], axis=1).plot.barh()
    plt.xlabel("delay")
    plt.title("Arrival delay per carrier during weekend and working days")
    plt.show()

    multi_index = [data["UNIQUE_CARRIER"], data["ORIGIN"], data["DEST"], data["FL_DATE"]]

    multi_index_df = data.set_index(multi_index)
    print((multi_index_df.loc[("AA", 'LAX')])[["DEP_TIME", "DEP_DELAY"]])
    print((multi_index_df.loc[("DL", 'LAX')])[["DEP_TIME", "DEP_DELAY"]])
    pass
    lax_arr = multi_index_df.xs("LAX", level=2, drop_level=False)
    lax_arr = lax_arr[lax_arr.FL_DATE < datetime.date(2017, 1, 8)]
    print(lax_arr.ARR_DELAY.mean())

    flights_per_day = (data.groupby(["weekday", "UNIQUE_CARRIER"]).FL_NUM.count()).unstack()
    sns.heatmap(flights_per_day.corr())
    plt.title("Flights per day by carrier (correlation)")
    plt.show()

    avg_delay_per_day = (data.groupby(["weekday", "UNIQUE_CARRIER"]).ARR_DELAY.mean()).unstack()
    sns.heatmap(avg_delay_per_day.corr())
    plt.title("Arrival delay per day by carrier (correlation)")
    plt.show()

    filtered = data[data.UNIQUE_CARRIER.isin(["HA", "DL", "AA", "AS"])]
    delta_per_day = filtered.groupby(["weekday", "UNIQUE_CARRIER"]).delaydelta.mean().unstack()

    delta_per_day.plot.line()
    plt.title("Delta delay per carrier")
    plt.show()
    pass


if __name__ == "__main__":
    main()
