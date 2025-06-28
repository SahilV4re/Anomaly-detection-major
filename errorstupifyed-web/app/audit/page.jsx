"use client";

import { useState } from "react";
import axios from "axios";
import { UploadCloud } from "lucide-react";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { SidebarInset, SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";
import AuditCard from "@/components/ui/auditcard";
import { Separator } from "@/components/ui/separator";
import {
    Breadcrumb,
    BreadcrumbItem,
    BreadcrumbList,
    BreadcrumbPage,
} from "@/components/ui/breadcrumb";
const groq_key = process.env.NEXT_PUBLIC_GROQ_API_KEY;

export default function UploadPage() {
    const [file, setFile] = useState(null);
    const [cards, setCards] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
        setError(null); // Clear previous errors when new file is selected
    };

    const parseCSV = (csvText) => {
        try {
            const [headerLine, ...lines] = csvText.trim().split("\n");
            const headers = headerLine.split(",").map(h => h.trim());

            return lines.filter(line => line.trim()).map((line) => {
                const values = line.split(",");
                const obj = {};
                headers.forEach((header, index) => {
                    obj[header] = values[index]?.trim() || "";
                });
                return obj;
            });
        } catch (err) {
            console.error("CSV parsing error:", err);
            throw new Error("Failed to parse CSV file. Please ensure it's properly formatted.");
        }
    };

    const generatePromptFromData = (data) => {
        if (!data || data.length === 0) return "";
        
        // Get sample of the data and column names
        const sampleRows = data.slice(0, 3);
        const columns = Object.keys(sampleRows[0] || {});
        
        return `Analyze this CSV data and generate a comprehensive data quality audit report in JSON format. 
        Include these sections: missing_values, outliers, duplicates, and inconsistent_formatting.
        For each issue, provide: column name, count/values, and suggested fixes.
        Sample data (first 3 rows): ${JSON.stringify(sampleRows)}
        Columns detected: ${columns.join(", ")}`;
    };

    const handleUpload = async () => {
        if (!file) {
            setError("Please select a CSV file.");
            return;
        }

        setIsLoading(true);
        setCards([]);
        setError(null);

        const reader = new FileReader();
        reader.onload = async (e) => {
            try {
                const csvText = e.target.result;
                const jsonData = parseCSV(csvText);
                
                if (jsonData.length === 0) {
                    throw new Error("CSV file appears to be empty or improperly formatted.");
                }

                const prompt = generatePromptFromData(jsonData);
                
                const response = await axios.post(
                    "https://api.groq.com/openai/v1/chat/completions", 
                    {
                        model: "llama3-70b-8192",
                        messages: [
                            {
                                role: "system",
                                content: `You are a Sales Analyser for detecting Anomalies in the business. Given any dataset provide 4 different anomalies and include reason of which records are the cause, and how to improve it Analyze the provided CSV data and return 
                                findings in this exact JSON format:
                                PROVIDE ALL 4 SECTIONS:
                                {
                                    "high_value_calculation": [{
                                        "column": string,
                                        "count": number,
                                        "suggested_fix": string
                                    }],
                                    "incorrect_tax_calculation": [{
                                        "column": string,
                                        "count": number,
                                        "suggested_fix": string
                                    }],
                                    "unapproved_discount": {
                                        "count": number,
                                        "suggested_fix": string
                                    },
                                    "cancellation_fraud": [{
                                        "column": string,
                                        "issue": string,
                                        "suggested_fix": string
                                    }]
                                }`
                            },
                            {
                                role: "user",
                                content: prompt
                            }
                        ],
                        response_format: { type: "json_object" },
                        temperature: 0.2 // Lower temperature for more consistent results
                    },
                    {
                        headers: {
                            "Content-Type": "application/json",
                            "Authorization": `Bearer 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`
                        },
                        timeout: 30000 // 30 seconds timeout
                    }
                );

                if (!response.data.choices?.[0]?.message?.content) {
                    throw new Error("Invalid response from AI service");
                }

                const analysis = JSON.parse(response.data.choices[0].message.content);
                const formattedCards = formatAnalysisToCards(analysis);
                setCards(formattedCards);
                
            } catch (error) {
                console.error("Error:", error);
                setError(error.message || "Failed to analyze CSV data. Please try again.");
                setCards(getSampleCards()); // Fallback to sample data
            } finally {
                setIsLoading(false);
            }
        };

        reader.onerror = () => {
            setError("Failed to read file. Please try again.");
            setIsLoading(false);
        };

        reader.readAsText(file);
    };

    const formatAnalysisToCards = (analysis) => {
        const cards = [];
        
        if (analysis.missing_values?.length > 0) {
            cards.push({
                title: "High Value Calculation",
                columns: ["Column", "Count", "Suggested Fix"],
                data: analysis.missing_values.map(item => ({
                    Column: item.column || "Unknown",
                    Count: item.count?.toString() || "N/A",
                    "Suggested Fix": item.suggested_fix || "Not specified"
                }))
            });
        }
        
        if (analysis.outliers?.length > 0) {
            cards.push({
                title: "Incorrect Tax Calculation",
                columns: ["Column", "Detected Values", "Suggested Fix"],
                data: analysis.outliers.map(item => ({
                    Column: item.column || "Unknown",
                    "Detected Values": item.count?.toString() || "N/A",
                    "Suggested Fix": item.suggested_fix || "Not specified"
                }))
            });
        }
        
        if (analysis.duplicates?.count > 0) {
            cards.push({
                title: "Unapproved Discount",
                columns: ["Rows Affected", "Suggested Fix"],
                data: [{
                    "Rows Affected": analysis.duplicates.count?.toString() || "N/A",
                    "Suggested Fix": analysis.duplicates.suggested_fix || "Not specified"
                }]
            });
        }
        
        if (analysis.inconsistent_formatting?.length > 0) {
            cards.push({
                title: "Cancellation Fraud",
                columns: ["Column", "Issue", "Suggested Fix"],
                data: analysis.inconsistent_formatting.map(item => ({
                    Column: item.column || "Unknown",
                    Issue: item.issue || "Formatting issue",
                    "Suggested Fix": item.suggested_fix || "Not specified"
                }))
            });
        }
        
        // If no issues found, show a success card
        if (cards.length === 0) {
            cards.push({
                title: "Data Quality Analysis",
                columns: ["Result"],
                data: [{
                    Result: "No significant data quality issues detected!"
                }]
            });
        }
        
        return cards;
    };

   
    return (
        <SidebarProvider>
            <AppSidebar />
            <SidebarInset>
                <header className="flex h-16 shrink-0 items-center gap-2 transition-[width,height] ease-linear group-has-data-[collapsible=icon]/sidebar-wrapper:h-12">
                    <div className="flex items-center gap-2 px-4">
                        <SidebarTrigger className="-ml-1" />
                        <Separator
                            orientation="vertical"
                            className="mr-2 data-[orientation=vertical]:h-4"
                        />
                        <Breadcrumb>
                            <BreadcrumbList>
                                <BreadcrumbItem>
                                    <BreadcrumbPage>Audit</BreadcrumbPage>
                                </BreadcrumbItem>
                            </BreadcrumbList>
                        </Breadcrumb>
                    </div>
                </header>

                <main className="flex flex-col h-[calc(100vh-4rem)] overflow-hidden">
                    <div className="w-full px-4 md:px-6 py-6 bg-white border-b border-gray-200">
                        <div className="w-full border-2 border-dashed border-gray-300 rounded-2xl p-8 flex flex-col items-center justify-center gap-4">
                            <UploadCloud className="w-10 h-10 text-gray-500" />
                            <p className="text-gray-600 text-lg font-medium">
                                Upload a CSV file to analyze data quality
                            </p>
                            <Input
                                type="file"
                                accept=".csv"
                                onChange={handleFileChange}
                                className="cursor-pointer bg-muted/50 text-gray-500"
                                disabled={isLoading}
                            />
                            <button
                                onClick={handleUpload}
                                disabled={isLoading || !file}
                                className="px-6 py-2 border bg-black border-black text-white rounded-md hover:bg-muted/50 hover:text-black transition disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                {isLoading ? "Analyzing..." : "Analyze CSV"}
                            </button>
                            {file && (
                                <p className="text-sm text-gray-500 mt-2">
                                    Selected file: <span className="font-medium">{file.name}</span>
                                    {isLoading && <span className="ml-2">(Processing...)</span>}
                                </p>
                            )}
                            {error && (
                                <p className="text-sm text-red-500 mt-2 text-center">
                                    {error}
                                </p>
                            )}
                        </div>
                    </div>

                    <ScrollArea className="flex-1 w-full px-4 md:px-6 py-6">
                        {isLoading ? (
                            <div className="w-full flex flex-col items-center justify-center py-12 gap-4">
                                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-gray-900"></div>
                                <p className="text-gray-600">Analyzing your CSV data...</p>
                            </div>
                        ) : (
                            <div className="w-full grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                               {cards.map((card, index) => (
    <div key={index} className=" border-none shadow-none bg-muted/50 rounded-xl shadow-md p-4 space-y-4 border">
        <h2 className="text-lg font-semibold text-gray-800">{card.title}</h2>
        <div className="overflow-x-auto">
            <table className="min-w-full text-sm text-left text-gray-600">
                <thead className="text-xs uppercase text-gray-500 border-b">
                    <tr>
                        {card.columns.map((col, idx) => (
                            <th key={idx} className="px-3 py-2 whitespace-nowrap">
                                {col}
                            </th>
                        ))}
                    </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                    {card.data.map((row, rowIdx) => (
                        <tr key={rowIdx} className="hover:bg-gray-50">
                            {card.columns.map((col, colIdx) => (
                                <td key={colIdx} className="px-3 py-2 whitespace-nowrap">
                                    {row[col] || "-"}
                                </td>
                            ))}
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    </div>
))}

                            </div>
                        )}
                    </ScrollArea>
                </main>
            </SidebarInset>
        </SidebarProvider>
    );
}