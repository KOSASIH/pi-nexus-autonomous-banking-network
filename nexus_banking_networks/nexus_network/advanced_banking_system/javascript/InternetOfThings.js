// InternetOfThings.js
import * as thingSpeak from 'thingspeak';

const channel = thingSpeak.channel(123456);
channel.writeField(1, 25.5);
channel.writeField(2, 45.6);
channel.writeFields([{ field1: 25.5, field2: 45.6 }]);
const data = channel.readFields(1, 2);
console.log(data);
