import React from 'react';
import Project from './Project';

const App = () => {
  const [projects, setProjects] = useState([]);

  useEffect(() => {
    // Get the projects from the contract
    impactInvestingContract.methods.getProjectIds().call().then((projectIds) => {
      const projects = [];
      for (let i = 0; i < projectIds.length; i++) {
        impactInvestingContract.methods.getProject(projectIds[i]).call().then((project) => {
          projects.push(project);
        });
      }
      setProjects(projects);
    });
  }, []);

  return (
    <div>
      <h1>Social Impact Investing</h1>
      <ul>
        {projects.map((project) => (
          <li key={project.id}>
            <Project project={project} />
          </li>
        ))}
      </ul>
    </div>
  );
};

export default App;
