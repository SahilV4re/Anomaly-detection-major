"use client";
import { AppSidebar } from "@/components/app-sidebar";
import { Badge } from "@/components/ui/badge";
import { useEffect, useState } from "react";
import animationData from '@/public/animation.json'; 
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import {
  Card,
  CardTitle,
  CardHeader,
  CardFooter,
  CardContent,
  CardDescription,
} from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import ScrollableTableCard from "@/components/ui/scrollable-table";
import { Car, Hash, TrendingUpIcon, TriangleAlert, Wallet } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  CartesianGrid,
  LineChart,
  Line,
} from "recharts";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import axios from "axios";
import Lottie from "lottie-react";


export default function Page() {
  const COLORS = [
    "#94a3b8", // slate-400
    "#64748b", // slate-500
    "#475569", // slate-600
    "#334155", // slate-700
  ];

  const columns = [
    "Date",
    "Invoice_No_",
    "Name",
    "Item_Name",
    "Qty_",
    "Final_Total",
    "Server_Name",
    "Table_No_",
    "Status",
    "Payment_Type",
    "Area",
    "Order_Type",
    "Category",
    "anomalies",
    "severity",
  ];

  const data = [
    {
      date: "2024-06-27",
      invoice_no_: "INV-001",
      name: "John Doe",
      item_name: "Pizza Margherita",
      qty_: 2,
      final_total: 500,
      server_name: "Alice",
      table_no_: "T3",
      status: "Completed",
      payment_type: "Card",
      area: "Ground Floor",
      order_type: "Dine-In",
      category: "Main Course",
      anomalies: "none",
      severity: "low",
    },
    {
      date: "2024-06-27",
      invoice_no_: "INV-001",
      name: "John Doe",
      item_name: "Pizza Margherita",
      qty_: 2,
      final_total: 500,
      server_name: "Alice",
      table_no_: "T3",
      status: "Completed",
      payment_type: "Card",
      area: "Ground Floor",
      order_type: "Dine-In",
      category: "Main Course",
      anomalies: "none",
      severity: "low",
    },

    {
      date: "2024-06-27",
      invoice_no_: "INV-001",
      name: "John Doe",
      item_name: "Pizza Margherita",
      qty_: 2,
      final_total: 500,
      server_name: "Alice",
      table_no_: "T3",
      status: "Completed",
      payment_type: "Card",
      area: "Ground Floor",
      order_type: "Dine-In",
      category: "Main Course",
      anomalies: "none",
      severity: "low",
    },
    {
      date: "2024-06-27",
      invoice_no_: "INV-001",
      name: "John Doe",
      item_name: "Pizza Margherita",
      qty_: 2,
      final_total: 500,
      server_name: "Alice",
      table_no_: "T3",
      status: "Completed",
      payment_type: "Card",
      area: "Ground Floor",
      order_type: "Dine-In",
      category: "Main Course",
      anomalies: "none",
      severity: "low",
    },
    {
      date: "2024-06-27",
      invoice_no_: "INV-001",
      name: "John Doe",
      item_name: "Pizza Margherita",
      qty_: 2,
      final_total: 500,
      server_name: "Alice",
      table_no_: "T3",
      status: "Completed",
      payment_type: "Card",
      area: "Ground Floor",
      order_type: "Dine-In",
      category: "Main Course",
      anomalies: "none",
      severity: "low",
    },
    {
      date: "2024-06-27",
      invoice_no_: "INV-001",
      name: "John Doe",
      item_name: "Pizza Margherita",
      qty_: 2,
      final_total: 500,
      server_name: "Alice",
      table_no_: "T3",
      status: "Completed",
      payment_type: "Card",
      area: "Ground Floor",
      order_type: "Dine-In",
      category: "Main Course",
      anomalies: "none",
      severity: "low",
    },
    {
      date: "2024-06-27",
      invoice_no_: "INV-001",
      name: "John Doe",
      item_name: "Pizza Margherita",
      qty_: 2,
      final_total: 500,
      server_name: "Alice",
      table_no_: "T3",
      status: "Completed",
      payment_type: "Card",
      area: "Ground Floor",
      order_type: "Dine-In",
      category: "Main Course",
      anomalies: "none",
      severity: "low",
    },

    // Add more rows as needed
  ];





  const filterData = {
    Methodology: ["Artificial Intelligence", "Manual"],
    Order_Type: ["Dine In", "Delivery(Parcel)", "Pick Up"],
    Payment_Type: [
      "CARD",
      "Cash",
      "Due Payment",
      "Online",
      "Other [AMEX]",
      "Other [Paytm]",
      "Other [ZOMATO PAY]",
      "Part Payment",
    ],
    Order_Source: ["Zomato", "Swiggy", "Local"],
    Status: ["Cancelled", "Complimentary", "Success"],
  };

  const [selectedFilters, setSelectedFilters] = useState({
    Order_Type: "Dine In",
    Payment_Type: null,
    Order_Source: null,
    Status: null,
    Methodology: "Artificial Intelligence",
  });

  const handleSelect = (category, value) => {
    setSelectedFilters((prev) => ({
      ...prev,
      [category]: prev[category] === value ? null : value,
    }));
    fetchData();
    
  };



  //API IMPLEMENTATION

  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [pieData, setPieData] = useState([]);
  const [barData, setBarData] = useState([]);
  const [line_data, setLineData] = useState([]);
  const [anamolyCategories, setAnomolyCategories] = useState([]);
  const SERVER_URL = process.env.NEXT_AI_URL;
  const RULE_BASED_URL = process.env.NEXT_RULE_BASED_URL;
  const fetchData = async () => {
    try {
      setLoading(true);
 

      var response;
      if (selectedFilters.Methodology === "Artificial Intelligence") {
        response = await axios.post(
          `http://localhost:8008/report`,
          {
            filters: Object.fromEntries(
              Object.entries({
                Order_Type: selectedFilters.Order_Type,
                Payment_Type: selectedFilters.Payment_Type,
                Order_Source: selectedFilters.Order_Source,
                Status: selectedFilters.Status,
                Methodology: selectedFilters.Methodology,
              })
                .filter(([_, value]) => value !== null)
                .map(([key, value]) => [key, [value]])
            ),
          },
          {
            headers: {
              "Content-Type": "application/json",
            },
          }
        );
      }
      else {
      const response = await axios.post(
        `http://localhost:8008/audit-report`,
        {
          filters: Object.fromEntries(
            Object.entries({
              Order_Type: selectedFilters.Order_Type,
              Payment_Type: selectedFilters.Payment_Type,
              Order_Source: selectedFilters.Order_Source,
              Status: selectedFilters.Status,
              Methodology: selectedFilters.Methodology,
            })
              .filter(([_, value]) => value !== null)
              .map(([key, value]) => [key, [value]])
          ),
        },
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );
    }
      if (response.status !== 200) {
        throw new Error("Failed to fetch data: " + response.data);
      }

      console.log(response.data);

      setPieData(
        Object.entries(response.data.visualization_data.anomaly_types).map(
          ([key, value]) => ({
            name: key,
            value: value,
          })
        )
      );

      setBarData(
        Object.entries(
          response.data.visualization_data.source_distribution
        ).map(([key, value]) => ({
          name: key,
          value: value,
        }))
      );

      setLineData(
        Object.entries(
          response.data.visualization_data.severity_distribution
        ).map(([key, value]) => ({
          name: key,
          users: value,
        }))
      );

      console.log("anomaly_categories:", response?.data?.anomaly_categories);


      setAnomolyCategories(
        Object.keys(
          response?.data?.anomaly_categories || {}
        )
        
       
        
      );

      // console.log(
      //   Object.keys(response.data.visualization_data.anomaly_categories)
      // );

      setReport(response.data);
      console.log("Anomaly categories:", anamolyCategories);


    } catch (error) {
      setError(error);
      console.error("Error fetching data:", error);
    } finally {
      setLoading(false);
    }
  };

    var selectedAnamolyCategory = anamolyCategories[0];

  useEffect(() => {
    console.log(SERVER_URL);
   
    fetchData();
  }, [selectedFilters]);

  if (loading) {
    return (
      <div className="w-full h-screen flex flex-col items-center justify-center">
      <Lottie animationData={animationData} loop autoplay />
      <div className="text-2xl font-semibold text-muted-foreground">
        Crunching the numbers...

        </div>
    </div>
    );
  }
  if (error) {
    return <div>Error: {error.message}</div>;
  }

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
                  <BreadcrumbPage>Dashboard</BreadcrumbPage>
                </BreadcrumbItem>
              </BreadcrumbList>
            </Breadcrumb>
          </div>
        </header>

        <div className="flex flex-1 flex-col gap-4 p-4 pt-0">
          <div className="flex flex-wrap gap-4 justify-start items-center w-auto border-none shadow-none ">
            <div className="text-sm font-semibold text-muted-foreground">
              {new Date().toLocaleString("en-IN", {
                dateStyle: "full",
                timeStyle: "short",
              })}
            </div>
            {Object.entries(filterData).map(([category, options]) => (
              <Popover key={category}>
                <PopoverTrigger asChild>
                  <Button
                    variant="outline"
                    className="capitalize px-3 py-1 text-sm bg-muted rounded-lg border border-muted hover:border-gray-300   hover:bg-white transition-all duration-200"
                  >
                    {category.replace(/_/g, " ")}{" "}
                    {selectedFilters[category]
                      ? `: ${selectedFilters[category]}`
                      : ""}
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-56">
                  <div className="flex flex-col gap-1">
                    {options.map((option) => (
                      <Button
                        key={option}
                        variant={
                          selectedFilters[category] === option
                            ? "default"
                            : "ghost"
                        }
                        className={cn(
                          "justify-start",
                          selectedFilters[category] === option && "font-bold"
                        )}
                        onClick={() => handleSelect(category, option)}
                      >
                        {option}
                      </Button>
                    ))}
                    {selectedFilters[category] && (
                      <>
                        <Separator className="my-2" />
                        <Button
                          onClick={() =>
                            handleSelect(category, selectedFilters[category])
                          }
                        >
                          Clear Filter
                        </Button>
                      </>
                    )}
                  </div>
                </PopoverContent>
              </Popover>
            ))}
          </div>

          <div className="grid auto-rows-min gap-4 md:grid-cols-3">
            <Card className="w-auto border-none shadow-none bg-muted/50">
              <CardHeader className="relative">
                <div className="flex justify-between items-center gap-2">
                  <div className="flex flex-col">
                    <CardDescription>Total Records</CardDescription>
                    <CardTitle className="card:text-3xl text-4xl font-semibold tabular-nums">
                      {report.total_stats.total_records}
                    </CardTitle>
                  </div>
                  <div className="rounded-3xl bg-primary p-5  flex items-center justify-center">
                    <Hash size={24} color="white" />
                  </div>
                </div>
              </CardHeader>
            </Card>
            <Card className="w-auto border-none shadow-none bg-muted/50">
              <CardHeader className="relative">
                <div className="flex justify-between items-center gap-2">
                  <div className="flex flex-col">
                    <CardDescription>Total Anomalies</CardDescription>
                    <CardTitle className="card:text-3xl text-4xl font-semibold tabular-nums">
                      {report.total_stats.total_anomalies}
                    </CardTitle>
                  </div>
                  <div className="rounded-3xl bg-primary p-5  flex items-center justify-center">
                    <TriangleAlert size={24} color="white" />
                  </div>
                </div>
              </CardHeader>
            </Card>
            <Card className="w-auto border-none shadow-none bg-muted/50">
              <CardHeader className="relative">
                <div className="flex justify-between items-center gap-2">
                  <div className="flex flex-col">
                    <CardDescription>Financial Impact</CardDescription>
                    <CardTitle className="card:text-3xl text-4xl font-semibold tabular-nums">
                      ₹ {report.total_stats.financial_impact.toLocaleString()}
                    </CardTitle>
                  </div>
                  <div className="rounded-3xl bg-primary p-5  flex items-center justify-center">
                    <Wallet size={24} color="white" />
                  </div>
                </div>
              </CardHeader>
            </Card>
          </div>

          <div className="w-full border-none shadow-none mt-2">
            <div className="flex flex-col md:flex-row gap-4">
              {/* Pie Chart */}
              <div className="w-full md:w-1/2">
                <div className="text-lg font-semibold mb-2">Anomaly Types</div>
                <ResponsiveContainer
                  width="100%"
                  height={300}
                  className="bg-muted/50 rounded-xl p-4"
                >
                  <PieChart>
                    <Pie
                      data={pieData}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={70}
                      label
                      isAnimationActive={true}
                      animationDuration={5000} // 1 second animation
                      animationEasing="ease-out" // optional easing
                    >
                      {pieData.map((entry, index) => (
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
              </div>

              {/* Bar Chart */}
              <div className="w-full md:w-1/2 ">
                <div className="text-lg font-semibold mb-2">
                  Source Distribution
                </div>
                <ResponsiveContainer
                  width="100%"
                  height={300}
                  className="bg-muted/50 rounded-xl p-4"
                >
                  <BarChart data={barData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Legend
                      color="000000"
                      formatter={(value) => (
                        <span style={{ color: "#000000" }}>{value}</span>
                      )}
                    />
                    <Bar dataKey="value">
                      {barData.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={COLORS[index % COLORS.length]}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
          <div></div>
          <div className="flex flex-col md:flex-row gap-4 w-full">
            <div className="text-lg font-semibold  flex-1 ">
              Anamoly Categories
            </div>
          </div>
          {/* 
          <Card className="bg-muted/50 rounded-xl border-none shadow-none">
       
        <CardContent> */}

        
          <div className="space-y-0">
            {/* Sample Business Inputs */}
            <div className="flex flex-wrap gap-2">
              {anamolyCategories.map((sample, index) => (
                <button
                  key={index}
                  onClick={(e) => {
             
                    selectedAnamolyCategory = sample;
                    // handleSubmit(e);
                  }}
                  className={
                    selectedAnamolyCategory === sample
                      ? "px-3 py-1 text-sm bg-white rounded-lg border-2 border-muted hover:border-gray-300   hover:bg-white transition-all duration-200"
                      : "px-3 py-1 text-sm bg-muted rounded-lg border border-muted hover:border-gray-300   hover:bg-white transition-all duration-200"
                  }
                >
                  {sample}
                </button>
              ))}
            </div>
          </div>
          {/* </CardContent>
      </Card> */}

          <ScrollableTableCard
          report={report.anomaly_categories[selectedAnamolyCategory]}
            title={selectedAnamolyCategory}
            columns={columns}
            data={data}
          />

          <Card className="w-auto border-none shadow-none bg-muted/50">
            <CardHeader>
              <CardTitle>Anomalies Count History</CardTitle>
            </CardHeader>
            <CardContent className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={line_data}
                  margin={{ top: 20, right: 30, left: 0, bottom: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Line
                    type="monotone"
                    dataKey="users"
                    stroke="#334155"
                    strokeWidth={2}
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
}
