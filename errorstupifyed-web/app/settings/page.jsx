'use client';
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";
import { useEffect, useState } from 'react';
import { Separator } from "@/components/ui/separator";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbPage,
  BreadcrumbList,
} from "@/components/ui/breadcrumb";
import Cookies from 'js-cookie';
import { useRouter } from 'next/navigation';
const SERVER_URL = process.env.NEXT_PUBLIC_API_URL;

const initialThresholds = {
  discountThreshold: 20,
  taxVarianceThreshold: 5,
  fraudCancellationRate: 10,
  cancellationMinutes: 15,
  highValueTransaction: 5000,
  repeatOrderThreshold: 3,
};

export default function SettingsPage() {
  const [thresholds, setThresholds] = useState(initialThresholds);
  const [uid, setUid] = useState(null);
  const router = useRouter();
  var cookieUid = "";

  useEffect(() => {
     cookieUid = Cookies.get("uid");
  }, []);

  const updateThreshold = (key, delta) => {
    setThresholds((prev) => ({
      ...prev,
      [key]: Math.max(0, prev[key] + delta),
    }));
  };

  const handleInputChange = (key, value) => {
    const num = parseInt(value, 10);
    if (!isNaN(num) && num >= 0) {
      setThresholds((prev) => ({
        ...prev,
        [key]: num,
      }));
    }
  };

  const handleSavePreferences = async () => {
    if (!cookieUid) return alert("UID not found. Please login again.");

    try {
      const response = await fetch(SERVER_URL+"/update_user", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          uid: cookieUid,
          update_data: thresholds,
        }),
      });

      const result = await response.json();
      if (response.ok) {
        alert("Preferences saved successfully!");
      } else {
        alert("Failed to save preferences: " + result.error);
      }
    } catch (err) {
      console.error(err);
      alert("Error saving preferences");
    }
  };

  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset>
        <header className="flex h-16 shrink-0 items-center gap-2 transition-[width,height] ease-linear group-has-data-[collapsible=icon]/sidebar-wrapper:h-12">
          <div className="flex items-center gap-2 px-4">
            <SidebarTrigger className="-ml-1" />
            <Separator orientation="vertical" className="mr-2 data-[orientation=vertical]:h-4" />
            <Breadcrumb>
              <BreadcrumbList>
                <BreadcrumbItem>
                  <BreadcrumbPage>Preferences</BreadcrumbPage>
                </BreadcrumbItem>
              </BreadcrumbList>
            </Breadcrumb>
          </div>
        </header>

        <div className="flex flex-1 flex-col gap-4 p-4 pt-0">
          <div className="">
            <div className="text-lg font-semibold flex-1 mb-4">
              Threshold Preferences for Rule-Based Analysis
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
              {Object.entries(thresholds).map(([key, value]) => (
                <div key={key} className="p-6 rounded-xl text-center border-none shadow-none bg-muted/50">
                  <h3 className="text-lg font-medium mb-4 capitalize">
                    {key.replace(/([A-Z])/g, ' $1')}
                  </h3>
                  <div className="flex items-center justify-center gap-4">
                    <button
                      onClick={() => updateThreshold(key, -1)}
                      className="w-10 h-10 text-lg bg-white border rounded-full hover:bg-gray-300"
                    >
                      −
                    </button>
                    <input
                      type="text"
                      value={value}
                      onChange={(e) => handleInputChange(key, e.target.value)}
                      className="w-20 text-center text-xl font-semibold border bg-white rounded-md py-1"
                      min="0"
                    />
                    <button
                      onClick={() => updateThreshold(key, 1)}
                      className="w-10 h-10 text-lg bg-white border rounded-full hover:bg-gray-300"
                    >
                      +
                    </button>
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-6 text-center">
              <button
                onClick={handleSavePreferences}
                className="bg-primary text-white px-6 py-2 rounded-lg shadow-md hover:bg-primary/90"
              >
                Save Preferences
              </button>
            </div>
          </div>
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
}
