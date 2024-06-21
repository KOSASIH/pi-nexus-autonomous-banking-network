// database-config.json.tpl
{
  "database": {
    "host": "{{ key "database/host" }}",
    "port": {{ key "database/port" | atoi }},
    "username": "{{ key "database/username" }}",
    "password": "{{ key "database/password" }}",
    "name": "{{ key "database/name" }}"
  }
}
