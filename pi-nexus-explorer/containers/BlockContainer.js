import React, { useState, useEffect } from 'react';
import { Block } from '../types';
import { apiUtils } from '../utils';
import BlockComponent from '../components/BlockComponent';

interface Props {
  match: {
    params: {
      blockNumber: string;
    };
  };
}

const BlockContainer: React.FC<Props> = ({ match }) => {
  const [block, setBlock] = useState<Block | null>(null);

  useEffect(() => {
    apiUtils.getBlock(match.params.blockNumber).then((block) => setBlock(block));
  }, [match.params.blockNumber]);

  return (
    <div>
      {block ? <BlockComponent block={block} /> : <p>Loading...</p>}
    </div>
  );
};

export default BlockContainer;
