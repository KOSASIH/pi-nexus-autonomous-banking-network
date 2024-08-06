import { Router } from 'express';
import { healthRecordRouter } from './healthRecord';
import { medicalBillingRouter } from './medicalBilling';
import { patientRouter } from './patient';
import { healthcareProviderRouter } from './healthcareProvider';

const apiRouter = Router();

apiRouter.use('/health-records', healthRecordRouter);
apiRouter.use('/medical-billing', medicalBillingRouter);
apiRouter.use('/patients', patientRouter);
apiRouter.use('/healthcare-providers', healthcareProviderRouter);

export { apiRouter };
