import time
from collections import defaultdict


class ModelJobBuffer:
    """
    Buffer to manage and group incoming jobs by their associated model.

    This class is designed to allow grouping of jobs that use the same model
    in order to process them together, while avoiding starvation for jobs
    using less common models by enforcing a maximum wait time.

    Attributes:
        max_wait_time (float): Maximum time (in seconds) to wait before 
                processing jobs of a model.
        jobs_by_model (defaultdict): Mapping of model_name -> list of job IDs.
        job_timestamps (dict): Mapping of job ID -> timestamp (float).
    """

    def __init__(self, max_wait_time=10):
        """
        Initializes the job buffer.

        Args:
            max_wait_time (float): Maximum waiting time in seconds for jobs 
                                    of the same model
                                   before processing. Default is 10 seconds.
        """
        self.jobs_by_model = defaultdict(list)
        self.job_timestamps = {}
        self.max_wait_time = max_wait_time


    def add_job(self, job_id, model_name):
        """
        Adds a job to the buffer.

        Args:
            job_id (str): Unique identifier of the job.
            model_name (str): Model associated with the job.
        """
        now = time.time()
        if job_id not in self.job_timestamps:
            self.jobs_by_model[model_name].append(job_id)
            self.job_timestamps[job_id] = now
            print(f"[BUFFER] Job {job_id} added to model {model_name}")


    def get_jobs_for_model(self, model_name):
        """
        Returns the list of job IDs associated with a given model.

        Args:
            model_name (str): The model name.

        Returns:
            list[str]: List of job IDs.
        """
        return self.jobs_by_model.get(model_name, [])


    def get_oldest_model(self):
        """
        Returns the model that has the oldest job in the buffer.

        Returns:
            str or None: The model name with the oldest job, or None if empty.
        """
        if not self.job_timestamps:
            return None

        oldest_job = min(self.job_timestamps.items(), key=lambda x: x[1])
        oldest_job_id = oldest_job[0]
        for model_name, jobs in self.jobs_by_model.items():
            if oldest_job_id in jobs:
                return model_name

        return None


    def ready_to_process(self, model_name):
        """
        Checks whether jobs for the given model are ready to be processed.
        A job is considered ready if its waiting time exceeds `max_wait_time`.

        Args:
            model_name (str): The model name.

        Returns:
            bool: True if the oldest job is ready to be processed.
        """
        jobs = self.jobs_by_model.get(model_name, [])
        if not jobs:
            return False
        first_job_id = jobs[0] # First one is the oldest
        timestamp = self.job_timestamps.get(first_job_id, 0)
        return (time.time() - timestamp) > self.max_wait_time


    def pop_jobs(self, model_name, max_jobs=None):
        """
        Pops and returns the oldest jobs associated with a given model.

        Args:
            model_name (str): The model name.
            max_jobs (int, optional): Max number of jobs to return. 
            If None, returns all.

        Returns:
            list[str]: List of job IDs popped from the buffer.
        """
        jobs = self.jobs_by_model.get(model_name, [])
        if max_jobs:
            jobs_to_return = jobs[:max_jobs]
            self.jobs_by_model[model_name] = jobs[max_jobs:]
        else:
            jobs_to_return = jobs
            self.jobs_by_model[model_name] = []

        for job_id in jobs_to_return:
            self.job_timestamps.pop(job_id, None)

        if not self.jobs_by_model[model_name]:
            del self.jobs_by_model[model_name]

        return jobs_to_return


    def remove_job(self, job_id):
        """
        Removes a job from the buffer.

        Args:
            job_id (str): The job ID to remove.
        """
        self.job_timestamps.pop(job_id, None)
        for model_name in list(self.jobs_by_model.keys()):
            if job_id in self.jobs_by_model[model_name]:
                self.jobs_by_model[model_name].remove(job_id)
                if not self.jobs_by_model[model_name]:
                    del self.jobs_by_model[model_name]
                #print(f"[BUFFER] Removed job {job_id} from model {model_name}")
                break


    def __str__(self):
        """
        Returns a string representation of the buffer state.

        Returns:
            str: Human-readable representation.
        """
        return f"[BUFFER STATE] {dict(self.jobs_by_model)}"

