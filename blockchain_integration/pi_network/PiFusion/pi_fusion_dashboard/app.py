from flask import Flask, render_template
from pi_fusion_dashboard.models import Node

app = Flask(__name__)

@app.route('/')
def index():
  nodes = Node.query.all()
  return render_template('dashboard.html', nodes=nodes)

if __name__ == '__main__':
  app.run(debug=True)
