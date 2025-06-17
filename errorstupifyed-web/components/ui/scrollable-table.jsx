"use client";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ScrollArea } from "@radix-ui/react-scroll-area";
import ReactMarkdown from "react-markdown";

export default function ScrollableTableCard({
  report,
  title = "Data Table",
  columns = [],
  data = [],
}) {
  const coreKeys = [
    "Invoice_No_",
    "Item_Name",
    "detected_flags",
    "model_flags",
    "severity",
    "severity_score",
    "model_anomaly_score",
  ];
  
  
  return (
    <div className="flex flex-row gap-4 w-full h-full">
      <Card className="w-auto border-none shadow-none bg-muted/50 flex-1 pb-2">
        <CardHeader className="relative">
          <CardDescription>{title}</CardDescription>
          <CardTitle className="card:text-3xl text-4xl font-semibold tabular-nums pb-0">
            {report['count']}
          </CardTitle>
        </CardHeader>
        <CardDescription>
          <ScrollArea className="max-h-64 w-full overflow-y-auto rounded-md  p-4 pt-0 pb-0">
            <ReactMarkdown>{report['recommendations']}</ReactMarkdown>
          </ScrollArea>
        </CardDescription>
      </Card>

      <ScrollArea className="w-full rounded-s-sm bg-background max-h-[600px] overflow-y-auto">
  <Card className="overflow-hidden border-none shadow-none bg-muted/50 p-0">
    <CardContent className="p-0">
      <div className="w-full overflow-x-auto">
        <table className="min-w-full text-sm text-left">
          <thead className="bg-muted text-muted-foreground">
            <tr>
              {coreKeys.map((key, index) => (
                <th key={index} className="px-4 py-2 font-medium">
                  {key}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {report.records.map((row, rowIndex) => (
              <tr
                key={rowIndex}
                className={
                  rowIndex % 2 === 0 ? "bg-background" : "bg-muted/30"
                }
              >
                {coreKeys.map((key, colIndex) => (
                  <td key={colIndex} className="px-4 py-2">
                    {row[key] !== null && row[key] !== undefined
                      ? String(row[key])
                      : "-"}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </CardContent>
  </Card>
</ScrollArea>


    </div>
  );
}
