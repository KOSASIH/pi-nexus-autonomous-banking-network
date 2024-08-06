// ipfs.js
const ipfs = require('ipfs-api')();

class IPFSStorage {
  async addFile(file) {
    const fileBuffer = Buffer.from(file, 'utf8');
    const fileAdded = await ipfs.add(fileBuffer);
    return fileAdded.hash;
  }

  async getFile(hash) {
    const fileBuffer = await ipfs.cat(hash);
    return fileBuffer.toString('utf8');
  }

  async addDirectory(directory) {
    const files = await Promise.all(
      directory.files.map(file => ipfs.add(file.buffer))
    );
    const directoryAdded = await ipfs.add({
      path: directory.path,
      mode: directory.mode,
      mtime: directory.mtime,
      files: files.map(file => ({ path: file.path, hash: file.hash }))
    });
    return directoryAdded.hash;
  }

  async getDirectory(hash) {
    const directory = await ipfs.get(hash);
    return directory.toJSON();
  }
}

module.exports = IPFSStorage;
