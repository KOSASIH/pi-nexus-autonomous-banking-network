import { PiBrowser } from '@pi-network/pi-browser-sdk';

const piBrowser = new PiBrowser({
  // Your API key here
  apiKey: 'YOUR_API_KEY',
});

// Configure the Pi Browser
piBrowser.configure({
  // Customize the browser's appearance
  theme: 'dark',
});

// Integrate the Pi Browser with your project
piBrowser.on('ready', () => {
  // Add the Pi Browser to your project's UI
  const piBrowserElement = piBrowser.getElement();
  document.body.appendChild(piBrowserElement);
});

export default piBrowser;
