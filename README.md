# QueryGrade

QueryGrade is a Django-based web application designed to analyze MySQL log files. It supports both slow query logs and general query logs, providing insights and identifying anomalies in the logs.

## Features

- Upload MySQL slow query logs and general query logs.
- Analyze the logs to identify anomalies.
- Display the results in a user-friendly web interface.

## Requirements

- Python 3.x
- Django 5.1.1
- pandas 2.2.3
- SQLAlchemy
- scikit-learn 1.5.2
- tensorflow
- torch
- sqlparse
- numpy 1.26.4

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/QueryGrade.git
    cd QueryGrade
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Apply migrations:
    ```bash
    python manage.py migrate
    ```

5. Run the development server:
    ```bash
    python manage.py runserver
    ```

6. Open your web browser and navigate to `http://127.0.0.1:8000/` to access the application.

## Usage

1. On the homepage, upload a MySQL log file (either slow query log or general query log).
2. Select the type of log file you are uploading.
3. Click the "Upload" button to analyze the log file.
4. View the results on the results page, which will display any anomalies found in the log file.

## Project Structure

```
manage.py
requirements.txt
analyzer/
    __init__.py
    admin.py
    apps.py
    forms.py
    models.py
    parser.py
    tests.py
    urls.py
    views.py
    migrations/
        __init__.py
    static/
        analyzer/
            css/
    templates/
        analyzer/
            base.html
            index.html
            results.html
media/
querygrade/
    __init__.py
    asgi.py
    settings.py
    urls.py
    wsgi.py
    __pycache__/
samples/
    mysql-general-query.log
    mysql-slow-query.log
static/
```


## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes or improvements.

## License

This project is licensed under the Apache 2.0 license. See the [LICENSE](LICENSE.md) file for details.