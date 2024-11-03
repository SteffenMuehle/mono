report:
	poetry run jupyter nbconvert --to html --execute src/finances.ipynb --output ../output/report_$(shell date +%Y-%m-%d)
prompt:
	poetry run python src/prompt_values.py