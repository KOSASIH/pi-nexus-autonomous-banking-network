// realtime_dashboard.js
const React = require('react')
const { useState, useEffect } = React

function RealtimeDashboard () {
  const [data, setData] = useState([])

  useEffect(() => {
    const dataIngestion = new DataIngestion()
    dataIngestion.ingestData().then((data) => {
      setData(data)
    })
  }, [])

  return (
    <div>
      <h1>Real-time Dashboard</h1>
      <ul>
        {data.map((datum, index) => (
          <li key={index}>
            {datum.timestamp}: {datum.value}
          </li>
        ))}
      </ul>
    </div>
  )
}
