// consul-template.hcl
template {
  source      = "/etc/consul-template.d/templates/api-config.json.tpl"
  destination = "/etc/api-config.json"
  perms       = 0600
  command     = "sudo service api restart"
}

template {
  source      = "/etc/consul-template.d/templates/database-config.json.tpl"
  destination = "/etc/database-config.json"
  perms       = 0600
  command     = "sudo service database restart"
}
