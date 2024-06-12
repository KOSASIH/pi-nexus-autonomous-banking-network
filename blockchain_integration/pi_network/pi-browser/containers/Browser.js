// containers/Browser.js
import React, { useState, useEffect } from 'react';
import { useSelector } from 'react-redux';
import { selectWalletAddress } from '../reducers/walletReducer';
import { compileCode } from '../api/codeCompiler';

const Browser = () => {
  const walletAddress = useSelector(selectWalletAddress);
  const [htmlCode, setHtmlCode] = useState('');
  const [cssCode, setCssCode] = useState('');
  const [jsCode, setJsCode] = useState('');
  const [compiledCode, setCompiledCode] = useState('');

  useEffect(() => {
    const compile = async () => {
      const code = await compileCode(htmlCode, cssCode, jsCode);
      setCompiledCode(code);
    };
    compile();
  }, [htmlCode, cssCode, jsCode]);

  const handleSave = () => {
    // Use Cordova's File plugin to save files to device storage
    const fileSystem = cordova.file.dataDirectory;
    const htmlFile = fileSystem + 'index.html';
    const cssFile = fileSystem + 'style.css';
    const jsFile = fileSystem + 'script.js';

    // Create files and write code to them
    fileSystem.getFile(htmlFile, { create: true }, (fileEntry) => {
      fileEntry.createWriter((writer) => {
        writer.write(htmlCode);
      });
    });

    fileSystem.getFile(cssFile, { create: true }, (fileEntry) => {
      fileEntry.createWriter((writer) => {
        writer.write(cssCode);
      });
    });

    fileSystem.getFile(jsFile, { create: true }, (fileEntry) => {
      fileEntry.createWriter((writer) => {
        writer.write(jsCode);
      });
    });
  };

  return (
    <div className="browser">
      <h1>Browser</h1>
      <textarea
        id="html"
        value={htmlCode}
        onChange={(e) => setHtmlCode(e.target.value)}
      />
      <textarea
        id="css"
        value={cssCode}
        onChange={(e) => setCssCode(e.target.value)}
      />
      <textarea
        id="js"
        value={jsCode}
        onChange={(e) => setJsCode(e.target.value)}
      />
      <button onClick={handleSave}>Save</button>
      <iframe srcDoc={compiledCode} />
    </div>
  );
};

export default Browser;
