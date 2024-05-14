const axios = require('axios')
const csvParser = require('csv-parser')
const fs = require('fs')

const dataSources = [
  {
    name: 'Source 1',
    url: 'https://example.com/data1.csv'
  },
  {
    name: 'Source 2',
    url: 'https://example.com/data2.csv'
  }
  // Add more data sources as needed
]

const ingestData = async () => {
  for (const source of dataSources) {
    try {
      const response = await axios.get(source.url, { responseType: 'stream' })
      const writer = fs.createWriteStream(`${source.name}.csv`)

      response.data.pipe(csvParser()).pipe(writer)

      console.log(`Data from ${source.name} ingested successfully.`)
    } catch (error) {
      console.error(
        `Failed to ingest data from ${source.name}:`,
        error.message
      )
    }
  }
}

ingestData()
