from django.test import TestCase
from django.urls import reverse
from .models import Job
from .serializers import JobSerializers

class JobModelTestCase(TestCase):
    def setUp(self):
        self.job = Job.objects.create(combinedparameters='test_parameters')

    def test_job_serializer(self):
        serializer = JobSerializers(instance=self.job)
        self.assertEqual(serializer.data, {'combinedparameters': 'test_parameters'})

class JobViewTestCase(TestCase):
    def setUp(self):
        self.job = Job.objects.create(combinedparameters='test_parameters')

    def test_job_list_create_view(self):
        url = reverse('job_list_create')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)

        data = {'combinedparameters': 'new_parameters'}
        response = self.client.post(url, data=data)
        self.assertEqual(response.status_code, 201)

    def test_job_retrieve_update_destroy_view(self):
        url = reverse('job_retrieve_update_destroy', args=[self.job.pk])
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)

        data = {'combinedparameters': 'updated_parameters'}
        response = self.client.put(url, data=data)
        self.assertEqual(response.status_code, 200)

        response = self.client.get(url)
        self.assertEqual(response.data['combinedparameters'], 'updated_parameters')

        response = self.client.delete(url)
        self.assertEqual(response.status_code, 204)
        self.assertFalse(Job.objects.filter(pk=self.job.pk).exists())
