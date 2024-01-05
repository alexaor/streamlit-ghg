.PHONY: run
run:
	poetry run streamlit run app.py


.PHONY: db
db:
	poetry run python init_vector_db.py
