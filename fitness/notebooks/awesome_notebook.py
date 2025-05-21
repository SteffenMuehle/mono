# %%
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from ipywidgets import widgets
from IPython.display import display

# %%
# load data
input_path = Path.cwd().parent / 'data'
file_name = "workouts.csv"
file_path = input_path / file_name
df = pd.read_csv(file_path)

# iso format and utc:
df["date"] = pd.to_datetime(df["date"], utc=True, format="ISO8601")
df["volume_summand"] = df["intensity"] * df["repetitions"]
df["plot_date"] = df["date"] + df.groupby(["date", "exercise"]).cumcount() * pd.Timedelta(hours=4)

# %%
df = pd.merge(
    left=df,
    right=df.groupby(["date", "exercise"]).agg(
        num_sets = pd.NamedAgg("volume_summand", "count"),
        volume = pd.NamedAgg("volume_summand", "sum"),
    ).reset_index(),
    on=["date", "exercise"],
    how="left",
)

# %%
# Widgets for user input
exercise_widget = widgets.Dropdown(
    options=df["exercise"].unique(),
    value=df["exercise"].unique()[0],
    description='Exercise:'
)

attribute_widget = widgets.Dropdown(
    options=["volume", "intensity", "repetitions", "num_sets"],
    value="volume",
    description='Attribute:'
)

# Assign an empty figure widget
trace = go.Scatter(x=[], y=[], mode='lines+markers', name='Exercise Data')
g = go.FigureWidget(data=[trace],
                    layout=go.Layout(
                        title=dict(
                            text='Exercise Data Over Time'
                        ),
                        xaxis=dict(title='Date'),
                        yaxis=dict(title='Value')
                    ))

# Function to update the plot based on widget input
def response(change):
    exercise = exercise_widget.value
    attribute = attribute_widget.value
    data = df[df["exercise"] == exercise]
    if attribute in ["num_sets","volume"]:
        data = data[ data["plot_date"] == data["date"] ]

    y_min = 0
    y_max = data[attribute].max() * 1.1
    
    with g.batch_update():
        g.data[0].x = data["plot_date"]
        g.data[0].y = data[attribute]
        g.layout.xaxis.title = 'Date'
        g.layout.yaxis.title = attribute
        g.layout.title = f'{attribute} over time for {exercise}'
        #g.layout.yaxis.range = [y_min, y_max]

# Observe changes in the widgets
exercise_widget.observe(response, names="value")
attribute_widget.observe(response, names="value")

# Initial plot display
response(None)

# Display the widgets and the plot
container = widgets.HBox([exercise_widget, attribute_widget])
display(widgets.VBox([container, g]))

# %%
