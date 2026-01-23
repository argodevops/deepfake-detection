import { useEffect, useMemo, useState } from "react";
import { Card } from "@/components/ui/card";
import { Brain, Shuffle, ScanFace, Repeat, Cpu, AlertTriangle } from "lucide-react";

type ModelInfoItem = {
  name: string;
  file: string;
  description: string;
  domain: string;
  output: "prob_real" | "prob_fake" | string;
  threshold: number;
  loaded: boolean;
};

type ModelsResponse = {
  models_dir: string;
  config_path: string;
  models_loaded: number;
  models: ModelInfoItem[];
};

function domainIcon(domain: string) {
  const d = (domain || "").toLowerCase();
  if (d.includes("synthetic")) return ScanFace;
  if (d.includes("reenact")) return Repeat;
  if (d.includes("manip")) return Shuffle;
  return Brain;
}

function prettyOutput(output: string) {
  if (output === "prob_real") return "Prob(REAL)";
  if (output === "prob_fake") return "Prob(FAKE)";
  return output;
}

export const ModelInfo = () => {
  const [data, setData] = useState<ModelsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadModels = async () => {
      try {
        const res = await fetch("/api/models");
        if (!res.ok) throw new Error(`Failed to load models (${res.status})`);
        const json: ModelsResponse = await res.json();
        setData(json);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unknown error");
      } finally {
        setLoading(false);
      }
    };

    loadModels();
  }, []);

  const loadedModels = useMemo(
    () => (data?.models ?? []).filter((m) => m.loaded),
    [data]
  );

  const hasSyntheticDetector = useMemo(
    () => loadedModels.some((m) => (m.domain || "").toLowerCase().includes("synthetic")),
    [loadedModels]
  );

  return (
    <Card className="p-6 animate-fade-in">
      <h3 className="text-xl font-semibold mb-1 flex items-center gap-2">
        <Cpu className="w-5 h-5 text-primary" />
        AI Models Used for Detection
      </h3>

      <div className="text-sm text-muted-foreground mb-4">
        {loading && "Loading model information…"}
        {!loading && error && (
          <span className="text-destructive">
            Failed to load model information: {error}
          </span>
        )}
        {!loading && !error && data && (
          <span>
            Loaded <span className="font-medium text-foreground">{data.models_loaded}</span>{" "}
            model{data.models_loaded === 1 ? "" : "s"}.
          </span>
        )}
      </div>

      {!loading && !error && data && (
        <>
          {!hasSyntheticDetector && (
            <div className="mb-4 flex gap-3 rounded-lg border border-yellow-500/30 bg-yellow-500/10 p-3">
              <AlertTriangle className="w-5 h-5 text-yellow-600 dark:text-yellow-400 mt-0.5" />
              <div className="text-sm">
                <div className="font-medium">Synthetic-face detector not loaded</div>
                <div className="text-muted-foreground">
                  Fully AI-generated GAN faces may be misclassified as REAL unless a{" "}
                  <span className="font-medium">synthetic_face</span> model is present.
                </div>
              </div>
            </div>
          )}

          <div className="space-y-4">
            {loadedModels.map((model) => {
              const Icon = domainIcon(model.domain);

              return (
                <div key={model.name} className="flex gap-4">
                  <div className="flex-shrink-0">
                    <div className="p-2 rounded-md bg-primary/10">
                      <Icon className="w-5 h-5 text-primary" />
                    </div>
                  </div>

                  <div className="min-w-0">
                    <div className="font-medium text-foreground break-words">
                      {model.name}
                    </div>

                    <div className="text-sm text-muted-foreground">
                      {model.description}
                    </div>

                    <div className="mt-2 flex flex-wrap gap-2 text-xs">
                      <span className="rounded-full border bg-muted/30 px-2 py-1 text-muted-foreground">
                        Domain:{" "}
                        <span className="font-medium text-foreground">
                          {(model.domain || "unknown").replaceAll("_", " ")}
                        </span>
                      </span>

                      <span className="rounded-full border bg-muted/30 px-2 py-1 text-muted-foreground">
                        Output:{" "}
                        <span className="font-medium text-foreground">
                          {prettyOutput(model.output)}
                        </span>
                      </span>

                      <span className="rounded-full border bg-muted/30 px-2 py-1 text-muted-foreground">
                        Threshold:{" "}
                        <span className="font-medium text-foreground">
                          {Number.isFinite(model.threshold) ? model.threshold.toFixed(2) : "—"}
                        </span>
                      </span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="mt-5 text-xs text-muted-foreground">
            Ensemble approach: if any model produces strong evidence of manipulation or synthesis
            (based on its configured threshold), the content is flagged.
          </div>
        </>
      )}
    </Card>
  );
};
