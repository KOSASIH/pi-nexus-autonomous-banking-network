# config/deploy.rb
set :application, "sidra_chain_integration"
set :repo_url, "https://github.com/KOSASIH/pi-nexus-autonomous-banking-network.git"

set :deploy_to, "/var/www/sidra_chain_integration"
set :deploy_via, :remote_cache

set :rails_env, "production"

set :sidekiq_role, :app
set :sidekiq_config, "config/sidekiq.yml"

namespace :deploy do
  desc "Restart application"
  task :restart do
    on roles(:app), in: :sequence, wait: 5 do
      execute :touch, release_path.join("tmp/restart.txt")
    end
  end

  after :restart, :clear_cache do
    on roles(:app), in: :sequence, wait: 5 do
      execute :rake, "cache:clear"
    end
  end
end
