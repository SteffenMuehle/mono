# RUN MONTHLY on 25th, i.e. before salary comes in

### 0. (OPTIONAL)

##### 0.0 Review giro + visa + Paypal Umsaetze

##### 0.1 reevaluate monthly grocery costs
- `poetry run python src/finances/grocery_cost.py`
- estimate monthly grocery costs basd on terminal output
- update value in `input/io.monthly.out.groceries`.

##### 0.1 review monthly expenses in io.toml
- did any other fixed costs change?
- `input/io.monthly.out.???`

###### 0.2 add Liz contributions in io.toml
- ask Wum if she wants to contribute. If so
    - add to `input/io.transaction.Elizabeth_contribution_????`
    - wait for money to show up in giro

##### 0.3 add major expenses in io.toml
`input/io.expense.????`


### run:

1. Transfer Verrechnungskonto to giro
2. Transfer giro to visa if visa<0
3. depot csv:
    - log into https://www.comdirect.com
    - go to depot
    - download .csv file (`export as Excel format`)
    - remove header and tail manually
    - copy to `input/depot/`
3. `just start`


# RUN YEARLY AFTER NEW YEAR'S EVE
- download Umsaetze in giro, verrechnung