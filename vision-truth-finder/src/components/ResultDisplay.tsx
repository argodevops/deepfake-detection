import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

type PerModelScore = {
  name: string;
  real_score: number;
};

type PredictionResult = {
  result: "REAL" | "FAKE" | "UNKNOWN";
  score: number;
  threshold: number;
  prediction_id: string;
  input_type: "image" | "video";
  trigger_model?: string | null;
  fake_confidence?: number | null;
  details?: string;
  per_model?: PerModelScore[];
};

export function ResultDisplay({
  result,
  onReset,
}: {
  result: PredictionResult;
  onReset: () => void;
}) {
  const verdict = result.result;
  const scorePct = Math.round((result.score ?? 0) * 100);
  const isFake = verdict === "FAKE";

  return (
    <Card className="p-6 animate-fade-in">
      <div
        className={`mb-4 rounded-lg border p-6 ${
          isFake
            ? "bg-destructive/10 border-destructive/30"
            : "bg-green-500/10 border-green-500/30"
        }`}
      >
        <div className="flex flex-col md:flex-row md:justify-center gap-10 text-center">
          <div className="flex flex-col items-center">
            <div className="h-5 text-sm uppercase tracking-wide text-muted-foreground">
              Detection Result
            </div>

            <div
              className={`text-4xl font-bold leading-none mt-1 ${
                isFake ? "text-destructive" : "text-green-600"
              }`}
            >
              {verdict}
            </div>
          </div>

          <div className="flex flex-col items-center">
            <div className="h-5 text-sm uppercase tracking-wide text-muted-foreground">
              Fake Confidence
            </div>

            <div
              className={`text-4xl font-bold leading-none mt-1 ${
                isFake ? "text-destructive" : "text-green-600"
              }`}
            >
              {scorePct}%
            </div>
          </div>
        </div>
      </div>


      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3 mb-4">
        <div className="text-sm text-muted-foreground">
          Prediction ID:{" "}
          <span className="font-mono text-foreground">{result.prediction_id}</span>
        </div>

        <Button variant="secondary" onClick={onReset}>
          Analyse another file
        </Button>
      </div>

      {result.details && (
        <p className="text-muted-foreground">{result.details}</p>
      )}

      {result.trigger_model && (
        <p className="mt-3 text-sm">
          <span className="text-muted-foreground">Triggered by model:</span>{" "}
          <span className="font-medium text-foreground">{result.trigger_model}</span>
          {typeof result.fake_confidence === "number" && (
            <span className="text-muted-foreground">
              {" "}
              ({Math.round(result.fake_confidence * 100)}% fake confidence)
            </span>
          )}
        </p>
      )}

      {result.per_model && result.per_model.length > 0 && (
        <div className="mt-6">
          <h4 className="text-lg font-semibold mb-3">Model breakdown</h4>
          <div className="overflow-auto rounded-lg border">
            <table className="w-full text-sm">
              <thead className="bg-muted/40">
                <tr>
                  <th className="text-left p-3">Model</th>
                  <th className="text-right p-3">Prob(REAL)</th>
                  <th className="text-right p-3">Prob(FAKE)</th>
                </tr>
              </thead>
              <tbody>
                {result.per_model.map((m) => {
                  const fake = 1 - m.real_score;
                  return (
                    <tr key={m.name} className="border-t">
                      <td className="p-3 font-medium">{m.name}</td>
                      <td className="p-3 text-right">
                        {(m.real_score * 100).toFixed(2)}%
                      </td>
                      <td className="p-3 text-right">
                        {(fake * 100).toFixed(2)}%
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </Card>
  );
}
