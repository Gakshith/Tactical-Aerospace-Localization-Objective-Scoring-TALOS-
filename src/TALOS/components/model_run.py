import cv2
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import deque
from scipy.optimize import linear_sum_assignment

class ModelRun:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mot = self.MultiObjectTracker(dt=1/30, device=self.device)
        self.engine = self.ThreatAssessmentEngine(origin=(0,0))
        self.threat_history_log = []

    class TrackerState(torch.nn.Module):
        def __init__(self, dt, device="cpu"):
            super().__init__()
            self.dt = dt
            self.F = torch.tensor([
                [1,0,dt,0,0,0], [0,1,0,dt,0,0], [0,0,1,0,0,0],
                [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]
            ], dtype=torch.float32).to(device)
            self.H = torch.tensor([
                [1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]
            ], dtype=torch.float32).to(device)

    class Tracker:
        def __init__(self, id, box, dt, device="cpu"):
            self.id = id
            self.dt = dt
            self.device = device
            self.state = ModelRun.TrackerState(dt, device=device)
            self.history = deque(maxlen=30)

            w, h = box[2] - box[0], box[3] - box[1]
            cx, cy = box[0] + w/2, box[1] + h/2
            self.x_hat = torch.tensor([cx, cy, 0, 0, w, h], dtype=torch.float32, device=device)
            self.P = torch.eye(6, device=device) * 10.0
            self.Q = torch.eye(6, device=device) * 0.1
            self.R = torch.eye(4, device=device) * 1.0

        def predict(self):
            self.x_hat = torch.matmul(self.state.F, self.x_hat)
            self.P = torch.matmul(torch.matmul(self.state.F, self.P), self.state.F.T) + self.Q

        def update(self, z):
            z = z.view(-1).to(self.device)
            S = torch.matmul(torch.matmul(self.state.H, self.P), self.state.H.T) + self.R
            K = torch.matmul(torch.matmul(self.P, self.state.H.T), torch.linalg.inv(S))
            y = z - torch.matmul(self.state.H, self.x_hat)
            self.x_hat += torch.matmul(K, y)
            self.P -= torch.matmul(K, torch.matmul(self.state.H, self.P))
            self.history.append((int(self.x_hat[0]), int(self.x_hat[1])))

        def get_bounding_box(self):
            cx, cy, _, _, w, h = self.x_hat.cpu().numpy()
            return [self.id, cx - w/2, cy - h/2, cx + w/2, cy + h/2]

    class ThreatAssessmentEngine:
        def __init__(self, origin=(0, 0)):
            self.origin = np.array(origin)
            self.tau = 5.0

        def compute_risk(self, tracker):
            state = tracker.x_hat.cpu().numpy()
            pos, vel = state[:2], state[2:4]
            P_trace = np.trace(tracker.P.cpu().numpy()[:2, :2])
            vec_to_origin = self.origin - pos
            dist = np.linalg.norm(vec_to_origin)
            unit_to_origin = vec_to_origin / (dist + 1e-6)
            v_close = np.dot(vel, unit_to_origin)

            urgency = np.exp(-(dist/v_close)/self.tau) if v_close > 0.5 else 0.0
            intent = np.clip(np.dot(vel/(np.linalg.norm(vel)+1e-6), unit_to_origin), 0, 1) if v_close > 0 else 0
            proximity = np.exp(-dist / 800)
            uncertainty_bonus = np.exp(-P_trace / 100)

            score = (0.4 * urgency) + (0.3 * proximity) + (0.2 * intent) + (0.1 * uncertainty_bonus)
            tti = round(dist/v_close, 1) if v_close > 0.5 else 999
            return round(float(score), 4), tti

    class MultiObjectTracker:
        def __init__(self, dt, device="cpu"):
            self.trackers = []
            self.next_id = 0
            self.dt = dt
            self.device = device

        def update(self, detections):
            if not detections: return

            processed_dets = []
            for d in detections:
                w, h = d[2]-d[0], d[3]-d[1]
                processed_dets.append([d[0]+w/2, d[1]+h/2, w, h])
            z_tensor = torch.tensor(processed_dets, dtype=torch.float32, device=self.device)

            if not self.trackers:
                for det in detections:
                    self.trackers.append(ModelRun.Tracker(self.next_id, det, self.dt, self.device))
                    self.next_id += 1
                return

            for t in self.trackers: t.predict()

            preds = torch.stack([t.x_hat[[0,1,4,5]] for t in self.trackers])
            cost_matrix = torch.cdist(preds, z_tensor).cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            matched_dets = set()
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 150:
                    self.trackers[r].update(z_tensor[c])
                    matched_dets.add(c)

            for i, det in enumerate(detections):
                if i not in matched_dets:
                    self.trackers.append(ModelRun.Tracker(self.next_id, det, self.dt, self.device))
                    self.next_id += 1

    def log_threat_data(self, threats, frame_count):
        for threat in threats[:5]:
            self.threat_history_log.append({
                'frame': frame_count,
                'id': threat['t'].id,
                'score': threat['score'],
                'tti': threat['tti']
            })

    def execute_run(self):
        MODEL_PATH = self.config.model_path
        VIDEO_PATH = self.config.source_video_path
        OUTPUT_PATH = self.config.output_video_path
        model = YOLO(MODEL_PATH)
        cap = cv2.VideoCapture(VIDEO_PATH)
        w, h, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
        out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            count += 1

            results = model.predict(frame, verbose=False, conf=0.5)
            detections = results[0].boxes.xyxy.tolist()

            self.mot.update(detections)

            if self.mot.trackers:
                threats = []
                for t in self.mot.trackers:
                    score, tti = self.engine.compute_risk(t)
                    threats.append({'t': t, 'score': score, 'tti': tti})

                threats.sort(key=lambda x: x['score'], reverse=True)

                if count % 30 == 0:
                    self.log_threat_data(threats, count)

                for i, threat in enumerate(threats):
                    t = threat['t']
                    _, x1, y1, x2, y2 = t.get_bounding_box()
                    color = (0, 0, 255) if i == 0 else (0, 255, 0)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, f"ID:{t.id} Risk:{threat['score']}", (int(x1), int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            out.write(frame)

        cap.release()
        out.release()

        if self.threat_history_log:
            pd.DataFrame(self.threat_history_log).to_csv("talos_threat_analysis.csv", index=False)
            print("Successfully saved threat log to CSV.")