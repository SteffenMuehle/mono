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


### 1. Update monthly statistics:
`poetry run python src/finances/monthly_io.py`


### 2. Set target in giro
`poetry run python src/finances/set_giro_target.py`


### 3. Update data from depot
1. log into https://www.comdirect.com
2. go to depot
3. download .csv file (`export as Excel format`)
4. remove header and tail manually
5. copy to `input/depot/`
6. `poetry run python src/finances/parse_depot.py`


### 4. Update crypto
`bash src/finances/crypto.sh`


### 5. Update other data
1. Transfer Verrechnungskonto to giro
2. Transfer giro to visa if visa<0
2. `poetry run python src/finances/prompt_values.py`


### 6. Run algorithm
`poetry run python src/finances/graphs.py`


### 7. git commit
`git commit -am "report 2024-10-25"; git push`

### 8. move money around


# RUN YEARLY AFTER NEW YEAR'S EVE
- download Umsaetze in giro, verrechnung