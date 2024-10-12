import pandas as pd
from pathlib import Path
import plotly.express as px

# load data
input_path = Path(__file__).parent.parent / 'input'
file_name = "workouts.csv"
file_path = input_path / file_name
df = pd.read_csv(file_path)
# iso format and utc:
df["date"] = pd.to_datetime(df["date"], utc=True, format="ISO8601")

print(df.columns)
print(df)

# plot
attribute = "repetitions"
for exercise in df["exercise"].unique():
    data = df[ df["exercise"]==exercise ]
    data["plot_date"] = data["date"] + data.groupby(["date","exercise"]).cumcount() * pd.Timedelta(hours=4)

    fig = px.line(data, x="plot_date", y=attribute, title=f"{attribute} over time", markers=True)

    # save plot as html
    output_path = Path(__file__).parent.parent / 'output'
    output_path.mkdir(exist_ok=True)
    output_file = output_path / f"{exercise}.html"
    fig.write_html(str(output_file))