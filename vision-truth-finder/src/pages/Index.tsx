
import { useState } from "react";
import { Card } from "@/components/ui/card";
import { FileUpload } from "@/components/FileUpload";
import { ResultDisplay } from "@/components/ResultDisplay";
import { ModelInfo } from "@/components/ModelInfo";

interface PerModelScore {
  name: string;
  real_score: number;
}

interface PredictionResult {
  result: "REAL" | "FAKE" | "UNKNOWN";
  score: number;
  threshold: number;
  prediction_id: string;
  input_type: "image" | "video";
  trigger_model?: string | null;
  fake_confidence?: number | null;
  details?: string;
  per_model?: PerModelScore[];
}

const Index = () => {
  const [isAnalysing, setIsAnalysing] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const handleFileUpload = async (file: File) => {
    setIsAnalysing(true);
    setResult(null);

    if (file.type.startsWith("image/")) {
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    } else {
      setPreviewUrl(null);
    }

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to analyze file");
      }

      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      setResult(data);
    } catch (error) {
      console.error("Error analyzing file:", error);
    } finally {
      setIsAnalysing(false);
    }
  };


  const resetFlow = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    setResult(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background">
      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-12 animate-fade-in">
          <h1 className="text-5xl font-bold md:leading-[1.35] bg-gradient-to-r from-primary to-primary/70 bg-clip-text text-transparent mb-4">
            Face Image Deepfake Detector
          </h1>
          <p className="text-xl text-muted-foreground mx-auto mb-6">
            Upload a face image to check whether it is real or has been altered using deepfake techniques.
          </p>
          <p className="text-sm text-muted-foreground">
            Currently only supports face images. Videos are not yet supported.
          </p>
        </div>
        <div className="max-w-4xl mx-auto">
          {!result && !isAnalysing && (
            <>
              <Card className="p-8 animate-scale-in">
                <FileUpload onFileUpload={handleFileUpload} />
              </Card>

              <div className="mt-6">
                <ModelInfo />
              </div>
            </>
          )}

          {previewUrl && (
            <Card className="p-4 animate-fade-in mb-6">
              <div className="flex justify-center">
                <img
                  src={previewUrl}
                  alt="Uploaded content"
                  className="max-h-[400px] w-auto rounded-lg border shadow-sm opacity-80"
                />
              </div>
            </Card>
          )}

          {isAnalysing && (
            <Card className="p-8 animate-fade-in">
              <div className="text-center">
                <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-primary mb-4"></div>
                <h3 className="text-xl font-semibold mb-2">Analysing Content</h3>
                <p className="text-muted-foreground">
                  Deepfake detection AI models are examining file for deepfake patterns...
                </p>
              </div>
            </Card>
          )}

          {result && 
            <ResultDisplay result={result} onReset={resetFlow} />
          }
        </div>
      </div>
    </div>
  );
};

export default Index;
