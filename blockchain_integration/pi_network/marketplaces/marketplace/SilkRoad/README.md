# Getting Started

To get started with Silkroad, follow these steps:

1. Clone the repository:

```bash

1. git clone https://github.com/KOSASIH/silkroad.git
```

2. Create a virtual environment and activate it:

```bash

1. python3 -m venv venv
2. source venv/bin/activate
```

3. Install the required packages:

```bash

1. pip install -r requirements.txt
```

Create the SQLite database:

```bash

1. python
2. >>> from silkroad import db
3. >>> db.create_all()
4. >>> exit()
```

4. Run the application:

```bash

1. flask run
```

Access the application at http://127.0.0.1:5000/

## Features

1. User registration and login
2. Product browsing and ordering
3. Order management
4. Token-based API for order management
5. API Documentation
The API documentation can be found at http://127.0.0.1:5000/apidocs/.
6. Security
Silkroad includes basic security measures, such as password hashing and token-based authentication. However, additional security measures should be implemented for a production environment.
7. Error Handling
Error handling is included in the application, but it should be expanded and customized for a production environment.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

Silkroad is released under the MIT License. See LICENSE for details.
