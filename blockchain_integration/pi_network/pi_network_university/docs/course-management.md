# Course Management
================

The Course Management system allows administrators to create, manage, and track courses.

## Features

* Course creation and editing
* Course categorization and tagging
* Course enrollment and tracking
* Course completion and certification

## API Endpoints
--------------

### Courses

* **GET /courses**: Retrieve a list of all courses
* **POST /courses**: Create a new course
* **GET /courses/:id**: Retrieve a specific course by ID
* **PUT /courses/:id**: Update a specific course by ID
* **DELETE /courses/:id**: Delete a specific course by ID

### Course Enrollments

* **GET /courses/:id/enrollments**: Retrieve a list of all enrollments for a specific course
* **POST /courses/:id/enrollments**: Enroll a user in a specific course
* **GET /courses/:id/enrollments/:user_id**: Retrieve a specific enrollment by user ID
* **PUT /courses/:id/enrollments/:user_id**: Update a specific enrollment by user ID
* **DELETE /courses/:id/enrollments/:user_id**: Delete a specific enrollment by user ID
