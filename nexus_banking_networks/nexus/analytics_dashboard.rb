require 'sinatra'
require 'd3js-ruby'

class AnalyticsDashboard < Sinatra::Base
    get '/' do
        @transactions = Transaction.all
        erb :index
    end

    get '/transactions' do
        @transactions = Transaction.all
        json @transactions
    end
end

__END__

@@ index
<!DOCTYPE html>
<html>
  <head>
    <title>Nexus Banking Network Analytics</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
  </head>
  <body>
    <h1>Nexus Banking Network Analytics</h1>
    <div id="chart"></div>
    <script>
      // Implement D3.js charting logic to display real-time analytics
    </script>
  </body>
</html>
