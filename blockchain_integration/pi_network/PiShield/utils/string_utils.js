class StringUtils {
    hexToString(hex) {
        return Buffer.from(hex, 'hex').toString('utf8');
    }

    stringToHex(str) {
        return Buffer.from(str, 'utf8').toString('hex');
    }
}

module.exports = StringUtils;
