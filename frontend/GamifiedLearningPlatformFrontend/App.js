import React from 'react';
import Lesson from './Lesson';
import Badge from './Badge';

const App = () => {
  const [lessons, setLessons] = useState([]);
  const [badges, setBadges] = useState([]);

  useEffect(() => {
    // Get the lessons and badges from the contract
    blockchainBasicsContract.methods .getLessons().call().then((lessons) => {
      setLessons(lessons);
    });
    blockchainBasicsContract.methods.getBadges().call().then((badges) => {
      setBadges(badges);
    });
  }, []);

  return (
    <div>
      <h1>Blockchain Basics</h1>
      <ul>
        {lessons.map((lesson) => (
          <li key={lesson.id}>
            <Lesson lesson={lesson} />
          </li>
        ))}
      </ul>
      <h1>Badges</h1>
      <ul>
        {badges.map((badge) => (
          <li key={badge.id}>
            <Badge badge={badge} />
          </li>
        ))}
      </ul>
    </div>
  );
};

export default App;
