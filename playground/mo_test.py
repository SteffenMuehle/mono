import marimo

__generated_with = "0.9.27"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __():
    x=2
    return (x,)


@app.cell
def __(x):
    print(x)
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
