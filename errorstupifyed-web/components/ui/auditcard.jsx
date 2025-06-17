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

export default function AuditCard({
  card,
}) {
  console.log(card);
  const markdown = `
Here are the requested recommendations for improving the Unapproved Complimentary Order Policy:\n\n**Root Cause Analysis:**\nThe root cause of this issue is the lack of a clear approval process for complimentary orders, leading to unapproved orders being processed without proper authorization. This may be due to inadequate training, insufficient policy documentation, or ineffective communication among staff members.\n\n**Improvement Steps:**\n\n1. **Establish a Clear Approval Process:** Develop a comprehensive approval process for complimentary orders, outlining the roles and responsibilities of staff members involved in the approval process. This should include designating specific approvers, setting up an approval workflow, and establishing clear guidelines for when complimentary orders can be granted.\n2. **Implement Automated Approval Workflows:** Utilize technology to automate the approval process, ensuring that all complimentary orders are routed to the designated approvers for review and approval. This can be achieved through the integration of approval workflows into the existing point-of-sale (POS) system or order management software.\n3. **Provide Ongoing Training and Communication:** Offer regular training sessions for staff members to ensure they understand the complimentary order policy and approval process. Additionally, establish open communication channels to address any questions or concerns staff members may have, ensuring that they feel empowered to follow the established process.\n\n**Expected Business Impact:**\nImplementing these improvement steps is expected to have a significant impact on the business, including:\n\n* Reduced financial losses due to unapproved complimentary orders\n* Improved compliance with company policies and procedures\n* Enhanced customer satisfaction through more efficient and accurate order processing\n* Increased staff confidence and empowerment through clear guidelines and training\n* Improved audit trails and reduced risk of fraudulent activities\n\nBy addressing the root cause of unapproved complimentary orders and implementing these improvement steps, the business can minimize financial losses, improve operational efficiency, and enhance customer satisfaction.
`;
  return (
    <div className="gap-4 w-full h-full">
      <Card className="w-auto border-none shadow-none bg-muted/50 flex-1 pb-2">
        <CardHeader className="relative">
          <CardDescription>{}</CardDescription>
          <CardTitle className="card:text-3xl text-4xl font-semibold tabular-nums pb-0">
            10
          </CardTitle>
        </CardHeader>
        <CardDescription>
          <ScrollArea className="max-h-64 w-full overflow-y-auto rounded-md  p-4 pt-0 pb-0">
            <ReactMarkdown>{card.suggested_fix}</ReactMarkdown>
          </ScrollArea>
        </CardDescription>
      </Card>

      <ScrollArea className="w-64 rounded-s-sm bg-background flex-3 max-h-fit overflow-y-auto"  >
        <Card className="overflow-hidden border-none shadow-none bg-muted/50 p-0 ">
          <CardContent className="p-0">
            <div className="w-full  overflow-x-auto">
              <table className="min-w-full text-sm text-left">
                <thead className="bg-muted text-muted-foreground">
                  <tr>
                    {columns.map((col, index) => (
                      <th key={index} className="px-4 py-2 font-medium">
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.map((row, rowIndex) => (
                    <tr
                      key={rowIndex}
                      className={
                        rowIndex % 2 === 0 ? "bg-background" : "bg-muted/30"
                      }
                    >
                      {columns.map((col, colIndex) => {
                        const key = col.toLowerCase().replace(/\s+/g, "");
                        return (
                          <td key={colIndex} className="px-4 py-2">
                            {row[key] ?? "-"}
                          </td>
                        );
                      })}
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
