// utils.js
const fs = require('fs');
const path = require('path');

/**
 * Get the current timestamp in milliseconds.
 * @returns {number} The current timestamp in milliseconds.
 */
function getTimestamp() {
  return Date.now();
}

/**
 * Read a file asynchronously.
 * @param {string} filePath - The path to the file to read.
 * @returns {Promise<string>} The contents of the file.
 */
async function readFile(filePath) {
  return fs.promises.readFile(filePath, 'utf8');
}

/**
 * Write a file asynchronously.
 * @param {string} filePath - The path to the file to write.
 * @param {string} contents - The contents to write to the file.
 * @returns {Promise<void>} A promise that resolves when the file has been written.
 */
async function writeFile(filePath, contents) {
  return fs.promises.writeFile(filePath, contents);
}

module.exports = { getTimestamp, readFile, writeFile };
