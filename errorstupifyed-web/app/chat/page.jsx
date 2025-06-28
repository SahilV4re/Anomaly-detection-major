"use client";

import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { SidebarInset, SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";
import { Send } from "lucide-react";
import { Separator } from "@/components/ui/separator";
import { ResponsiveContainer, PieChart, Pie, Cell, Tooltip, Legend, BarChart, CartesianGrid, XAxis, YAxis, Bar, LineChart, Line } from "recharts";
import ReactMarkdown from "react-markdown";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import axios from "axios";
const COLORS = [
  "#94a3b8", // slate-400
  "#64748b", // slate-500
  "#475569", // slate-600
  "#334155", // slate-700
];

export default function ChatPage() {
  const [messages, setMessages] = useState([
    { role: "assistant", text: "Hello! I'm your AI assistant. How can I help you today?" },
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = { role: "user", text: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsTyping(true);

    try {
      const res = await axios.post("http://127.0.0.1:8008/chat", {
        message: input,
      });

      const assistantMessage = {
        role: "assistant",
        text: res.data.response || "No response received.",
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Error sending message:", error);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: "Sorry, there was an error getting a response." },
      ]);
    } finally {
      setIsTyping(false);
    }
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
                  <BreadcrumbPage>Chat</BreadcrumbPage>
                </BreadcrumbItem>
              </BreadcrumbList>
            </Breadcrumb>
          </div>
        </header>

        <main className="flex flex-col h-[calc(100vh-4rem)]">
          <ScrollArea className="flex-1 px-4 md:px-6 py-6">
            <div className="flex flex-col gap-6 max-w-3xl mx-auto">
              {messages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`flex w-full ${
                    msg.role === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  <div
                    className={`rounded-2xl px-6 py-4 max-w-2xl shadow-sm ${
                      msg.role === "user"
                        ? "bg-black text-white"
                        : "bg-white text-gray-800 border border-gray-200"
                    }`}
                  >
                    <div className="text-base leading-relaxed">
                
                     {ChatMessage(msg={msg})}
                    
                    </div>
                  </div>
                </div>
              ))}

              {isTyping && (
                <div className="flex justify-start">
                  <div className="rounded-2xl px-6 py-4 bg-white text-gray-800 border border-gray-200 shadow-sm">
                    <div className="flex gap-1">
                      <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: "0ms" }}></div>
                      <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: "200ms" }}></div>
                      <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: "400ms" }}></div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>

          <footer className="border-t border-gray-200 bg-white p-4 md:p-6">
            <div className="max-w-3xl mx-auto">
              <div className="flex gap-2 items-center bg-white rounded-lg border border-gray-300 focus-within:ring-1  focus-within:border-transparent p-1">
                <Input
                  className="flex-1 border-none shadow-none focus-visible:ring-0 focus-visible:ring-offset-0 text-base"
                  placeholder="Message Cognify..."
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSend()}
                />
                <Button 
                  onClick={handleSend}
                  size="icon"
                  className="bg-black hover:bg-gray-800 text-white h-10 w-10 rounded-lg"
                >
                  <Send className="h-5 w-5" />
                </Button>
              </div>
            </div>
          </footer>
        </main>
      </SidebarInset>
    </SidebarProvider>
  );

  
}

const ChatMessage = ({ msg }) => {
  let parsedData = null;

  try {
    if (msg && msg.text && msg.text[0] === "{" && msg.text.includes("}")) {
      const jsonStart = msg.text.indexOf("{");
      const jsonEnd = msg.text.lastIndexOf("}");
      const jsonString = msg.text.slice(jsonStart, jsonEnd + 1);
      const json = JSON.parse(jsonString);
  
      if (json.chart_type && json.data && Array.isArray(json.data)) {
        parsedData = json;
      }
    }
  } catch (err) {
    console.error("Failed to parse chart JSON:", err);
  }

  if (parsedData) {
    const { chart_type, data } = parsedData;

    switch (chart_type) {
      case "PieChart":
        return (
          <ResponsiveContainer
            width="100%"
            height={300}
            className="bg-muted/50 rounded-xl p-4"
          >
            <PieChart>
              <Pie
                data={data}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                outerRadius={70}
                label
                isAnimationActive={true}
                animationDuration={1000}
                animationEasing="ease-out"
              >
                {data.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={COLORS[index % COLORS.length]}
                  />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        );

      case "BarChart":
        return (
          <ResponsiveContainer
            width="100%"
            height={300}
            className="bg-muted/50 rounded-xl p-4"
          >
            <BarChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend
                formatter={(value) => (
                  <span style={{ color: "#000000" }}>{value}</span>
                )}
              />
              <Bar dataKey="value">
                {data.map((value, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={COLORS[index % COLORS.length]}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        );

      case "LineChart":
        return (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={data} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="value"
                stroke="#334155"
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        );

      default:
        return <div className="text-base leading-relaxed text-red-500">Unsupported chart type</div>;
    }
  }

  // Default fallback: render Markdown
  return (
    <div className="text-base leading-relaxed">
      <ReactMarkdown>{msg.text}</ReactMarkdown>
    </div>
  );
};

