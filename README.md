# CIANNA ON THE FLY

## Introduction 

The CIANNA On the fly (CIANNA-OTF) is a client–server framework designed to streamline the execution of AI-based inference tasks in astronomy and related fields.
Its main objective is to provide researchers and engineers with a standardized, scalable, and automated way to submit jobs, process large scientific datasets, and retrieve results without requiring direct access to the underlying infrastructure.


### Why CIANNA OTF?

1. Abstraction of Complexity


Users do not need to manually configure GPUs, models, or file structures.
Job submission is handled via simple REST or UWS requests, making the system accessible even to non-experts in system administration.

2. Reproducibility & Traceability

Every job is described by a parameters.xml file, ensuring consistent inputs and metadata.
The server validates, logs, and tracks the entire job lifecycle from submission to results.

3. Resource Management

The scheduler automatically allocates available CPU/GPU resources according to job requirements and hardware availability.
This allows multiple users and processes to share the same infrastructure efficiently.

4. Error Handling & Robustness

The system provides clear error reporting (e.g., missing model definitions, invalid parameters, GPU out-of-memory).
Logs and metrics are systematically stored for debugging and performance monitoring.

5. Scalability & Extensibility

New models and pipelines (e.g., YOLO + CIANNA, diffusion models, custom post-processing) can be integrated without altering the client interface.
The architecture is designed to support additional backends, hardware profiles, and data formats.

### Typical Use Case

A researcher submits an astronomical image (e.g., FITS from a radio survey) together with metadata (RA, DEC, image size, target model).
The client packages and sends the request.
The server validates the job, schedules resources, runs inference through CIANNA/YOLO models, and generates outputs (detections, annotations, logs).
The results are then retrieved by the client in structured formats (JSON/CSV, annotated images, archives).
This workflow makes CIANNA OTS a reliable bridge between advanced AI pipelines and the practical needs of scientists handling massive datasets.


## 1) High-Level architecture


- `JOBS/PENDING/<job_id>/` — uploaded files live here until scheduled.

- `JOBS/EXECUTING/<job_id>/` — during preprocessing/forward.

- `JOBS/COMPLETED/<job_id>/fwd_res/net0_rts_<id>.dat`\ — final output.

- `\JOBS/ERROR/<job_id>/`\ — failed jobs with status updated in log.

Following the UWS status define in the documention. 

## 2) Request lifecycle (UWS‑style)

1. Create job: client uploads parameters.xml + image.fits to POST /jobs/.

2. Job receipt: server creates job_id, writes to JOBS/PENDING/<id>/, logs the job, returns 303 Location: /jobs/<id>.

3. Background scheduling:

    - scheduler_loop watches JOBS/PENDING and feeds an in‑memory per‑model queue.
    - monitor_batch_buffers snapshots queues when size ≥ threshold or wait time ≥ MAX_WAIT_TIME.

4. Batch inference:

     - Preprocess (FITS normalize + tiling) per job.
    - Concatenate patches → single CIANNA forward → .dat tensor.
    - Slice outputs per job, write fwd_res/net0_rts_<id>.dat.

5. Finalize:

    - Update status to COMPLETED or ERROR.
    - Move job dir from EXECUTING → {COMPLETED|ERROR}.

6. Retrieve:

    - GET /jobs/<id> returns phase JSON or XML. -> **Plan VOTable instead**.
    - GET /jobs/<id>/results downloads .dat when COMPLETED **-> Should disapear** 

## 3) API surface (Flask)

### POST /jobs/

    - Form‑data: xml=<parameters.xml>, fits=<image.fits>.

    - Response: 303 See Other with Location: /jobs/<job_id>.

### GET /jobs/<job_id>

- Response: { jobId, phase, timestamp } (JSON) or UWS XML if Accept: application/xml.

### GET /jobs/<job_id>/results

- Response: attachment fwd_res/net0_rts_<job_id>.dat when phase COMPLETED.

## 4) Modules and responsibilities
### 4.1 main.py — Web API & server lifecycle

- Exposes endpoints above.
- Starts two background threads on boot:
    - `scheduler_loop(poll_interval, max_wait_time)`
    - `monitor_batch_buffers(poll_interval=1.0)`
- Utilities: job directory lookup, UWS phase mapping, XML builder for status.

### 4.2 pipeline.py — Core orchestration & batch inference

- Queueing: per‑model in‑memory queues with enqueue_job, try_snapshot_batch, mark_batch_done.
- Model lifecycle: global lock + ACTIVE_MODEL to avoid repeated loads.
- Preprocess: normalize_and_patch (normalization → tiles).
- Batch inference: batch_prediction (stack patches → CIANNA → dispatch outputs → move job to destination).
- Batch monitor: monitor_batch_buffers (trigger batch by size or time).
- Post‑processing (WIP): pred2csv skeleton (NMS, thresholds).

### 4.3 process_xml.py — Parameters parsing & registry

- `parse_job_parameters`: robust UWS parser (namespace tolerant) → returns (params_client, params_model).

- `get_model_info`: resolve a model by id from CIANNA_models.xml or UWS‑style file.

- Legacy helpers kept for compatibility: parse_job_parameters_xml, parse_job_parameters_old.

### 4.4 job_scheduler.py — Pending → queues

- Scans `JOBS/PENDING` for jobs.
- Extracts model identifier → adds the job to the shared per‑model buffer.
- (Legacy) Alternative priority‑based scheduler is present but not currently wired to the new queue.

## 5) Key functions (signatures & role)

### API

`create_job()` → handles upload & job creation; starts a thread to run_prediction_job.

`get_job_status(job_id)` → reports UWS phase.

`get_job_results(job_id)` → streams .dat output when ready.

### Orchestration

`run_prediction_job(process_id, xml_path, fits_path, model_dir)` → parse params, ensure model loaded once, create EXECUTING dir, enqueue job in per‑model queue.

`monitor_batch_buffers(poll_interval)` → periodically snapshot queues and launch batch_prediction.

`batch_prediction(jobs, model_path, base_params, cianna)` → preprocess per job, unified forward, dispatch to per‑job files, update statuses.

### Preprocess

`normalize_image(data, methods, scale, clip_min, clip_max)` → chained transforms (clip/linear/sqrt/log/tanh).

`normalize_and_patch(fits_path, params_client, params_model)` → read FITS, normalize, tile, return X, n_patches, and img_info.

### XML & registry

`parse_job_parameters(xml_path)` → tolerant UWS parse, merges byReference; mirrors FITS_Path in params_client.

`get_model_info(xml_path, model_id)` → resolve model metadata (dimensions, quantization, etc.).

## 6) Configuration (ServerConfig)

Minimum keys used across modules (names may differ by section):

- Paths: jobs_pending, jobs_executing, jobs_completed, jobs_error, models_dir, model_registry_path, runtime_root.

- Batch: batch_jobs_thres, max_wait_time.

- Server: server.port, server.verbose.

**TODO**: consolidate naming and ensure params/server.toml is the single source of truth.


