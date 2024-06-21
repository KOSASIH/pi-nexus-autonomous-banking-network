// api-config.json.tpl
{
  "api": {
    "host": "{{ key "api/host" }}",
    "port": {{ key "api/port" | atoi }},
    "timeout": {{ key "api/timeout" | atoi }}
  },
  "database": {
    "host": "{{ key "database/host" }}",
    "port": {{ key "database/port" | atoi }},
    "username": "{{ key "database/username" }}",
    "password": "{{ key "database/password" }}"
  }
}
